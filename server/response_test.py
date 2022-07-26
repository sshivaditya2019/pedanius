import requests


#Cassava Brown Streak Disease
print("Teacher")
resp = requests.post("http://localhost:5000/predict_teacher",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/2036664947.jpg','rb')})
print(resp.json())
print("Student")
resp = requests.post("http://localhost:5000/predict_student",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/2036664947.jpg','rb')})
print(resp.json())

#Cassava Mosaic Disease
print("Teacher")
resp = requests.post("http://localhost:5000/predict_teacher",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/1858241102.jpg','rb')})
print(resp.json())
print("Student")
resp = requests.post("http://localhost:5000/predict_student",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/1858241102.jpg','rb')})
print(resp.json())


#
print("Teacher")
resp = requests.post("http://localhost:5000/predict_teacher",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/3108176484.jpg','rb')})
print(resp.json())
print("Student")
resp = requests.post("http://localhost:5000/predict_student",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/3108176484.jpg','rb')})
print(resp.json())


#Healthy
# /home/sshivaditya/Projects/pedanius/data/train_images/19890423.jpg
print("Teacher")
resp = requests.post("http://localhost:5000/predict_teacher",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/19890423.jpg','rb')})
print(resp.json())
print("Student")
resp = requests.post("http://localhost:5000/predict_student",files={"file": open('/home/sshivaditya/Projects/pedanius/data/train_images/19890423.jpg','rb')})
print(resp.json())