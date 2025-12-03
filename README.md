# YOLO Lotto Server

## สิ่งที่ต้องมี 
ไฟล์ `.env`
## วิธีรัน (Docker)
1) Start Postgres (เริ่มจาก cd เข้าไปใน โฟลเดอร์ repo)
```
docker compose build  # ครั้งแรก หรือเมื่อแก้โค้ด/requirements
```

2) Build + run OCR server 
```
docker compose up -d
```

3) Verify containers
Expect 2 containers running:
```
docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                    NAMES
...            yolo-server2   "python ws_ocr_serve…"   ... seconds ago   Up ... seconds  0.0.0.0:8765->8765/tcp   yolo-server2
...            postgres:15    "docker-entrypoint.s…"   ... seconds ago   Up ... seconds  0.0.0.0:5432->5432/tcp   pg-lotto
```

4) Prepare database (e.g., via DBeaver)
- Create table:
```
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
```

5) Run client
```
python client.py
```

## Notes
- `.env` is ignored from Git
- ถ้าอยากบังคับใช้ CPU ให้ตั้ง `YOLO_DEVICE=cpu` ใน `.env`

