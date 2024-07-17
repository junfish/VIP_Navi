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
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
my_posenet_model =  model_parser("Resnet34", False, True, 0.1, False)
my_posenet_model = my_posenet_model.to(device)
model_path = "/home/juy220/PycharmProjects/VIP_Navi/models_Resnet34_Basement-2024-05-10-00:57:23/best_net.pth"
my_posenet_model.load_state_dict(torch.load(model_path))
my_posenet_model.eval()

# Define image transformation pipeline

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

# files = sorted(glob.glob("/home/juy220/PycharmProjects/VIP_Navi/navip_server/frame*"))
files = sorted(glob.glob("/home/juy220/PycharmProjects/VIP_Navi/navip_server/frame_*"))
frame_count = 0
overlay = map_image.copy()
for file in files:
    frame_count += 30
    img = Image.open(file)
    pos_out, ori_out = predict_image(img)
    prediction_info = f'Frame {frame_count}: Position--{pos_out}, Orientation--{ori_out}\n'
    print(prediction_info)
    cv2.circle(overlay, (int(pos_out[1]), int(pos_out[0])), radius=4, color=(0, 0, 255), thickness=-1)
    # Display the updated image map
    # img_path = cv2.addWeighted(overlay, 0.5, map_image, 1 - 0.5, 0)
    cv2.imshow('Image Map', overlay)
    cv2.waitKey(1)

cv2.imwrite('path.png', overlay)



