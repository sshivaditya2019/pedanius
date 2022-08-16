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
import timm
import os
os.path.dirname(os.path.dirname(__file__))
from model import net
import utils
app = Flask(__name__)
print("App Started")
imagenet_class_index = json.load(open('/home/sshivaditya/Projects/pedanius/data/label_num_to_disease_map.json'))

class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''
    def forward(self, x):
        x = self.model(x)
        return x



model_ft = CassvaImgClassifier("tf_efficientnet_b4_ns",5,pretrained=True)
model_ft.load_state_dict(torch.load("/home/sshivaditya/Projects/pedanius/saves/CrossEntropy/tf_efficientnet_b4_ns_fold_0_9"))
model_ft.eval()
json_path = os.path.join("configs/",'params.json')
assert os.path.isfile(json_path)
params = utils.Params(json_path)
model_kd = net.Net(params=params)
checkpoint = torch.load("/home/sshivaditya/Projects/pedanius/saves/best.pth.tar")
model_kd.load_state_dict(checkpoint['state_dict'])
model_kd.eval()

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

def transform_image_v2(image_bytes):
    my_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomResizedCrop(256),
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

def get_prediction_v2(image_bytes):
    tensor = transform_image_v2(image_bytes=image_bytes)
    outputs = model_kd.forward(tensor)    
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx,imagenet_class_index[predicted_idx]

@app.route('/predict_teacher', methods=['POST'])
def predict_teacher():
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

@app.route('/predict_student', methods=['POST'])
def predict_student():
    if request.method == 'POST':
        tracemalloc.start()
        st = time.time()
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction_v2(image_bytes=img_bytes)
        mem = tracemalloc.get_traced_memory()
        et = time.time()
        tracemalloc.stop()
        return jsonify({'class_id': class_id, 'class_name': class_name, 'Memory': mem, 'Time': et-st})

if __name__ == '__main__':
    app.run()