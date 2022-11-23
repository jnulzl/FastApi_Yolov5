''' Example client sending POST request to server and printing the YOLO results
'''
import sys
import requests as r
import numpy as np
import json


def binary2file(content, file_path):
    with open(file_path, "wb") as fpR:
        fpR.write(content)


def send_request(image):
    # http --form POST http://10.78.3.128:8080/face-detection image@images/1.jpg
    res = r.post(url="http://10.78.9.128:8000/detection",
                    files={'image': open(image, "rb")})
    # binary2file(res.content, "res.jpg")
    # img_buffer = np.frombuffer(res.content, dtype=np.uint8)
    # img_buffer.tofile("ddd.jpg")
    # print(type(res.raw), type(res.content), len(res.content))
    # print(res, res.text, type(res.text))
    # print(res, res.content, type(res.content))
    print(res, res.json(), type(res.json()))


if __name__ == '__main__':
    image = sys.argv[1]
    send_request(image)