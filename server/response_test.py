import requests

#Healthy
resp = requests.post("http://localhost:5000/predict",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/1858350755.jpg','rb')})
print(resp.json())

# Csassava Mosaic Disease
#resp = requests.post("http://localhost:5000/predict",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/1858241102.jpg','rb')})
#print(resp.json())


