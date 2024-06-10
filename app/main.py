from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import io
import picamera
import time
import websockets
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with picamera.PiCamera() as camera:
        camera.start_preview()
        # Camera warm-up time
        time.sleep(2)
        stream = io.BytesIO()
        for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):

            await websocket.send_bytes(stream.getvalue())
            # reset stream for next frame
            stream.seek(0)
            stream.truncate()
