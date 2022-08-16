import requests
""" import neptune.new as neptune
import glob
run = neptune.init(
        project="sshivaditya/cassava",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNjVmNmI1ZS0zN2Y4LTQzMDgtYTk1Yy03NzdjMzgzNzVjYTIifQ==",
        name="KDTest"
    ) """

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

""" go = glob.glob("/home/sshivaditya/Projects/pedanius/data/train_images/*.jpg")
for a in go:
    resp = requests.post("http://localhost:5000/predict_teacher",files={"file": open(a,'rb')})
    resp = resp.json()
    run["KDTest_Teacher/memory_avg"].log(resp["Memory"][0])
    run["KDTest_Teacher/memory_peak"].log(resp["Memory"][1])
    run["KDTest_Teacher/time"].log(resp["Time"])
    resp = requests.post("http://localhost:5000/predict_student",files={"file": open(a,'rb')})
    resp = resp.json()
    run["KDTest_student/memory_avg"].log(resp["Memory"][0])
    run["KDTest_student/memory_peak"].log(resp["Memory"][1])
    run["KDTest_student/time"].log(resp["Time"]) """