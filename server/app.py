import io
import json
import tracemalloc
import time
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import torch.nn as nn

app = Flask(__name__)
imagenet_class_index = json.load(open('/home/sshivaditya/Projects/pedanius/data/label_num_to_disease_map.json'))
model_ft = models.resnext50_32x4d(pretrained= True)
num_ftrs = model_ft.fc.in_features
model_ft.fc= nn.Linear(num_ftrs,5)
model_ft.load_state_dict(torch.load("/home/sshivaditya/Projects/pedanius/saves/FirstModel"))
model_ft.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees = 45),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model_ft.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx,imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tracemalloc.start()
        st = time.time()
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        mem = tracemalloc.get_traced_memory()
        et = time.time()
        tracemalloc.stop()
        return jsonify({'class_id': class_id, 'class_name': class_name, 'Memory': mem, 'Time': et-st})


if __name__ == '__main__':
    app.run()