import asyncio, websockets, json, base64
img_b64 = base64.b64encode(open("pic/26_grayscale.jpg","rb").read()).decode()
async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(json.dumps({"image_base64": img_b64}))
        print(await ws.recv())
asyncio.run(main())

