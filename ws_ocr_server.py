import base64
import cv2
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta
import asyncio
import websockets
import psycopg2
from dotenv import load_dotenv

# ⬇️ IMPORT ฟังก์ชัน OCR เดิมของฟอล์ค
from read_lotto_hybrid_super import read_ticket_hybrid_fast

# Load environment variables for DB config
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

db_conn = None


def db_is_configured():
    return all([DB_CONFIG.get("dbname"), DB_CONFIG.get("user"), DB_CONFIG.get("password")])


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lotto_results (
              id SERIAL PRIMARY KEY,
              date_unix BIGINT NOT NULL,
              digit1 CHAR(1) NOT NULL,
              digit2 CHAR(1) NOT NULL,
              digit3 CHAR(1) NOT NULL,
              digit4 CHAR(1) NOT NULL,
              digit5 CHAR(1) NOT NULL,
              digit6 CHAR(1) NOT NULL
            );
            """
        )


def get_db_conn():
    global db_conn
    if db_conn and not db_conn.closed:
        return db_conn
    if not db_is_configured():
        return None
    db_conn = psycopg2.connect(**DB_CONFIG)
    db_conn.autocommit = True
    ensure_table(db_conn)
    return db_conn


def insert_numbers(date_unix, numbers):
    """Insert list of 6-digit strings into lotto_results."""
    conn = get_db_conn()
    if not conn:
        return
    rows = []
    for n in numbers:
        if len(n) == 6:
            rows.append((date_unix, n[0], n[1], n[2], n[3], n[4], n[5]))
    if not rows:
        return
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO lotto_results
                (date_unix, digit1, digit2, digit3, digit4, digit5, digit6)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                rows
            )
    except Exception as e:
        print(f"[DB] Insert failed: {e}")


# ---------------------------------------------------------
# Helper: Convert base64 → OpenCV grayscale image
# ---------------------------------------------------------
def base64_to_image(b64_string):
    try:
        image_bytes = base64.b64decode(b64_string)
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        return img
    except:
        return None


# ---------------------------------------------------------
# Main Websocket Handler
# ---------------------------------------------------------
async def ocr_handler(websocket):

    async for message in websocket:
        try:
            request = json.loads(message)

            if "image_base64" not in request:
                await websocket.send(json.dumps({"error": "Missing image_base64"}))
                continue

            # Decode image
            img_gray = base64_to_image(request["image_base64"])
            if img_gray is None:
                await websocket.send(json.dumps({"error": "Cannot decode image_base64"}))
                continue

            # =====================================================
            # Run OCR (ฟังก์ชัน OCR ของฟอล์ค 그대로)
            # =====================================================
            yolo_device = os.getenv("YOLO_DEVICE", "cpu").lower()

            detections, numbers, annotated, times, invert_used = read_ticket_hybrid_fast(
                image_path=None,image_array=img_gray,                # ไม่ใช้ path เพราะเรามีรูปใน memory
                model_path="lotto2.pt",
                conf_threshold=0.3,
                method="threshold",
                resize_scale=3,
                padding=5,
                invert="yes",
                threshold_method="binary",
                adaptive_block=20,
                adaptive_c=2,
                binary_thresh=150,
                save_crops=False,
                debug_dir="debug",
                calibrate_lines=True,        # เปิดหมุน
                calibrate_threshold_method='adaptive_gaussian',
                calibrate_adaptive_block=15,
                calibrate_adaptive_c=3,
                calibrate_canny_low=50,
                calibrate_canny_high=150,
                calibrate_hough_thresh=80,
                calibrate_min_line_len=50,
                calibrate_max_line_gap=10,
                device=yolo_device,   # เปลี่ยนเป็น "cpu" ถ้าไม่มี GPU
                mp_threshold=30
            )

            # =====================================================
            # Filter → Only 6-digit numbers
            # =====================================================
            valid_numbers = []
            for txt in numbers:
                digits = ''.join(c for c in txt if c.isdigit())
                if len(digits) == 6:
                    valid_numbers.append(digits)

            # =====================================================
            # Performance = valid / total detections
            # =====================================================
            performance = f"{len(valid_numbers)}/{detections}"

            # =====================================================
            # JSON Output Response
            # =====================================================
            th_tz = timezone(timedelta(hours=7))  # Thailand is UTC+7
            date_unix = int(datetime.now(th_tz).timestamp())

            # Optional DB insert
            try:
                insert_numbers(date_unix, valid_numbers)
            except Exception as db_err:
                print(f"[DB] Error while inserting: {db_err}")

            response = {
                "DATE_UNIX": date_unix,
                "Performance": performance,
                "numbers": valid_numbers
            }

            await websocket.send(json.dumps(response, ensure_ascii=False))

        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))


# ---------------------------------------------------------
# Start WebSocket Server
# ---------------------------------------------------------
async def main():
    print("🔥 WebSocket OCR Server started at ws://0.0.0.0:8765")
    async with websockets.serve(ocr_handler, "0.0.0.0", 8765, max_size=10_000_000):
        await asyncio.Future()     # Run forever


if __name__ == "__main__":
    asyncio.run(main())
