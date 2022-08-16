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
import student_model
import teacher_model
from teacher_model.engine import get_net as gn
from teacher_model import config 
from student_model import config as cf
from student_model.engine import get_net as gns
os.path.dirname(os.path.dirname(__file__))


app = Flask(__name__)
print("App Started")
imagenet_class_index = json.load(open('/home/sshivaditya/Projects/pedanius/data/label_num_to_disease_map.json'))




teacher_mod = gn(name=cf.TEACHER_NAME, pretrained=config.PRETRAINED)
teacher_mod.load_state_dict(torch.load(cf.PATH_TO_TEACHER))
teacher_mod.eval()


student_mod = gns(name=cf.NET, pretrained=cf.PRETRAINED)
student_mod.load_state_dict(torch.load("/home/sshivaditya/Projects/CKD/generated/weights/cnn/cnn_fold_3_14.bin"))
student_mod.eval()


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
    outputs = teacher_mod(tensor)    
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx,imagenet_class_index[predicted_idx]

def get_prediction_v2(image_bytes):
    tensor = transform_image_v2(image_bytes=image_bytes)
    outputs = student_mod(tensor)    
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