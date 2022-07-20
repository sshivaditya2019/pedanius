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

app = Flask(__name__)
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