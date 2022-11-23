import os
import sys
from typing import List, Tuple, Dict
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import uvicorn

import cv2
import numpy as np
import time

import pyobjdet as yolov5


weights_path = "face_320_one_output_fp16.mnn110"

yolov5_config = yolov5.YoloConfig()

yolov5_config.input_names = ["input"]
yolov5_config.output_names = ["output"]
yolov5_config.weights_path = weights_path;
yolov5_config.deploy_path = "";

yolov5_config.means[0] = 0;
yolov5_config.means[1] = 0;
yolov5_config.means[2] = 0;

yolov5_config.scales[0] = 0.0039215;
yolov5_config.scales[1] = 0.0039215;
yolov5_config.scales[2] = 0.0039215;

yolov5_config.mean_length = 3;
yolov5_config.net_inp_channels = 3;

yolov5_config.net_inp_width = 320;
yolov5_config.net_inp_height = yolov5_config.net_inp_width;

yolov5_config.num_threads = 2;

yolov5_config.num_cls = 1;
yolov5_config.conf_thres = 0.4;
yolov5_config.nms_thresh = 0.5;

yolov5_config.strides = [8, 16, 32];
yolov5_config.anchor_grids = [[10, 13, 16, 30, 33, 23],[30, 61, 62, 45, 59, 119],[116, 90, 156, 198, 373, 326]];


yolov5_obj = yolov5.PyYoloV5("mnn")


app = FastAPI(title="Jnulzl",
            swagger_ui_parameters={"defaultModelsExpandDepth": -1})


class Faces(BaseModel):
    faces: List
    

def detRun(img):
    begin_time = time.time()
    yolov5_obj.process(img, yolov5.IMG_BGR)
    rects = yolov5_obj.get_result()
    end_time = time.time()
    eps = (end_time - begin_time) * 1000
    print("Average time is %fms"%(eps))
    if len(rects) > 0:
        faces = [{'xmin':int(rect.x1), 'ymin':int(rect.y1),
                  'xmax':int(rect.x2), 'ymax':int(rect.y2)} for rect in rects]
        face_out = Faces(faces=faces)
    return faces
    
    
@app.post("/detection", response_model=Faces)
async def detection(image: UploadFile = File(...)) -> Faces:
    data = np.fromfile(image.file, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    faces = detRun(img)
    return Faces(faces=faces)
    
    
@app.post("/detection_show")
async def detection_show(image: UploadFile = File(...)):
    data = np.fromfile(image.file, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    faces = detRun(img)
    for face in faces:
        pt1 = (face['xmin'], face['ymin'])
        pt2 = (face['xmax'], face['ymax'])
        color = (0, 255, 255)
        thickness = 2
        cv2.rectangle(img, pt1, pt2, color, thickness)
    res, im_show = cv2.imencode(".jpg", img)
    return StreamingResponse(BytesIO(im_show.tobytes()), media_type="image/jpg")
    

@app.on_event("startup")
async def startup():
    yolov5_obj.init(yolov5_config)
    print("Init success!")
    
    
@app.on_event("shutdown")
async def shutdown():    
    print("\nDestroy success!\n")


if __name__ == "__main__":
    uvicorn.run("fastapi_demo_det:app", host="0.0.0.0", port=8000, log_level="info")