# YOLO Lotto Server

## Quickstart (Docker)
1) Start Postgres
```
docker run -d --name pg-lotto -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=adminlotto -e POSTGRES_DB=lotto_sub_number -p 5432:5432 postgres:15
```

2) Build + run OCR server (เริ่มจาก cd เข้าไปใน โฟลเดอร์ repo)
```
docker build -t yolo-server2 .
docker run -d --name yolo-server2 -p 8765:8765 --env-file .env -e YOLO_DEVICE=cpu yolo-server2
```

3) Verify containers
Expect two containers running:
```
docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                    NAMES
...            yolo-server2   "python ws_ocr_serve…"   ... seconds ago   Up ... seconds  0.0.0.0:8765->8765/tcp   yolo-server2
...            postgres:15    "docker-entrypoint.s…"   ... seconds ago   Up ... seconds  0.0.0.0:5432->5432/tcp   pg-lotto
```

4) Prepare database (e.g., via DBeaver)
- Connect to Postgres using credentials from `.env` (ส่งให้ในไลน์)
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
- Set `YOLO_DEVICE=cpu` to force CPU inference;
