import requests

#Cassava Brown Streak Disease
resp = requests.post("http://localhost:5000/predict",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/2036664947.jpg','rb')})
print(resp.json())

#Cassava Mosaic Disease
resp = requests.post("http://localhost:5000/predict",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/1858241102.jpg','rb')})
print(resp.json())

#
resp = requests.post("http://localhost:5000/predict",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/3108176484.jpg','rb')})
print(resp.json())

#Healthy
# /home/sshivaditya/Projects/pedanius/data/train_images/19890423.jpg
resp = requests.post("http://localhost:5000/predict",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/19890423.jpg','rb')})
print(resp.json())