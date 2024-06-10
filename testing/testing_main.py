import os

import cv2
import numpy as np
import asyncio
import websockets
from PIL import Image
from io import BytesIO
import pytest
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
load_dotenv()
# Check if a GPU is available and if not, use a CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
model.eval()

        # Define the transformation
transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((384,384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
server = os.getenv('SERVER')
port = os.getenv('PORT')
@pytest.mark.asyncio
async def test_websocket():
    uri = f"ws://{server}:{port}/ws"  # Adjust the URI as needed
    async with websockets.connect(uri, ping_timeout=600) as websocket:
        # Test the WebSocket connection
        assert websocket.open
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        mp_draw = mp.solutions.drawing_utils
        process_hands = None
        # Initialize the VideoWriter

        # Continuously receive and display images
        try:
            initial = True
            while True:
                # Test receiving data (image)
                image_bytes = await websocket.recv()
                image = Image.open(BytesIO(image_bytes))
                frame = np.array(image)
                if initial:
                    height, width, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or use 'XVID'
                    out = cv2.VideoWriter('output.mp4', fourcc, 20.0,
                                          (width, height))  # adjust the filename, codec, fps, and frame size as needed
                    initial = False

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                if process_hands is not None:
                    # Process the image and draw landmarks
                    results = hands.process(rgb_image)
                    if results.multi_hand_landmarks:  # If any hands are detected
                        for hand_landmarks in results.multi_hand_landmarks:  # For each hand
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw hand landmarks

                    # Write the frame to the VideoWriter# Write the frame to the file
                    out.write(frame)


                    # Display the image
                    cv2.imshow('Video Stream', frame)

                # Apply the transformation to the image
                img_tensor = transform(rgb_image).unsqueeze(0)
                # Move the input tensor to the same device as the model
                img_tensor = img_tensor.to(device)
                # Estimate the depth
                with torch.no_grad():
                    depth = model(img_tensor)
                # Convert the depth to a numpy array
                depth_np = depth.squeeze().cpu().numpy()

                # Normalize the depth map to the range 0-255
                depth_np = cv2.normalize(depth_np, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Resize the depth map to the size of the input frame
                depth_np = cv2.resize(depth_np, (frame.shape[1], frame.shape[0]))

                # Apply a bilateral filter to the depth map
                depth_np = cv2.bilateralFilter(depth_np, 9, 75, 75)

                # Apply histogram equalization
                depth_np = cv2.equalizeHist(depth_np)

                # Apply a colormap for better visualization
                depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
                # Display the depth map
                cv2.imshow('Depth Map', depth_np)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        finally:
            out.release()
            cv2.destroyAllWindows()



if __name__ == '__main__':
    test_websocket()
    print('Test completed successfully.')