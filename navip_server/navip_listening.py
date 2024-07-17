import socket
import av
import threading
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import model_parser
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
my_posenet_model =  model_parser("Resnet34", False, False, 0.0, False)
my_posenet_model = my_posenet_model.to(device)
model_path = "/home/juy220/PycharmProjects/VIP_Navi/models_Resnet34_Basement-2024-05-10-00:57:23/best_net.pth"
my_posenet_model.load_state_dict(torch.load(model_path))
my_posenet_model.eval()

# Define image transformation pipeline
# transform = transforms.Compose([
#             transforms.Resize(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

r = 3
red_bgr = (128, 128, 240)

# Load predefined image map
map_image = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_BASEMENT_low.jpeg')


def predict_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        pos_out, ori_out, _ = my_posenet_model(image)
        pos_out = pos_out.squeeze(0).detach().cpu().numpy()
        ori_out = F.normalize(ori_out, p=2, dim=1)
        ori_out = ori_out.squeeze(0).detach().cpu().numpy()
    return pos_out, ori_out



def handle_client_connection(client_socket):
    try:
        # Initialize PyAV
        container = av.open(client_socket.makefile(mode='rb'), format='h264')

        frame_count = 0
        temp_map_image = map_image.copy()

        for frame in container.decode(video=0):
            frame_count += 1
            # Get dimensions of the frame
            # width, height = frame.width, frame.height

            # Optionally save every 30th frame as an image
            if frame_count % 30 == 0:
                # Convert frame to PIL image
                img = frame.to_image()  # Converts to PIL Image
                # Get prediction from the DNN model
                pos_out, ori_out = predict_image(img)
                prediction_info = f'Frame {frame_count}: Position--{pos_out}, Orientation--{ori_out}\n'
                img.save(f'frame_{frame_count:04}.png')
                print(prediction_info)

                # Clone the map image to ensure it's unaffected by previous modifications
                cv2.circle(temp_map_image, (int(float(pos_out[1])), int(float(pos_out[0]))), radius=5, color=(0, 0, 255), thickness=-1)

                # Display the updated image map
                cv2.imshow('Image Map', temp_map_image)
                cv2.waitKey(1)

                # Send prediction info back to client
                client_socket.sendall(prediction_info.encode('utf-8'))


    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()

def start_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f'Server listening on {host}:{port}')
    while True:
        client_sock, address = server_socket.accept()
        print(f'Accepted connection from {address[0]}:{address[1]}')
        client_handler = threading.Thread(
            target=handle_client_connection,
            args=(client_sock,)
        )
        client_handler.start()

if __name__ == '__main__':
    start_server('0.0.0.0', 12005)
