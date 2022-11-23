import cv2
import sys
import numpy as np
import pyobjdet as yolov5
import time

if 4 != len(sys.argv):
    print("Usage:\n\t %s img_list.txt model_path input_size\n"%(sys.argv[0]))
    exit(0)

weights_path = sys.argv[2]
deploy_path = sys.argv[2]

yolov5_config = yolov5.YoloConfig()

yolov5_config.input_names = ["input"]
yolov5_config.output_names = ["output"]
yolov5_config.weights_path = weights_path;
yolov5_config.deploy_path = deploy_path;

yolov5_config.means[0] = 0;
yolov5_config.means[1] = 0;
yolov5_config.means[2] = 0;

yolov5_config.scales[0] = 0.0039215;
yolov5_config.scales[1] = 0.0039215;
yolov5_config.scales[2] = 0.0039215;

yolov5_config.mean_length = 3;
yolov5_config.net_inp_channels = 3;

yolov5_config.net_inp_width = int(sys.argv[3]);
yolov5_config.net_inp_height = yolov5_config.net_inp_width;

yolov5_config.num_threads = 2;

yolov5_config.num_cls = 1;
yolov5_config.conf_thres = 0.4;
yolov5_config.nms_thresh = 0.5;

yolov5_config.strides = [8, 16, 32];
yolov5_config.anchor_grids = [[10, 13, 16, 30, 33, 23],[30, 61, 62, 45, 59, 119],[116, 90, 156, 198, 373, 326]];


yolov5_obj = yolov5.PyYoloV5("mnn")
yolov5_obj.init(yolov5_config)

with open(sys.argv[1], "r") as fpR:
    lines = fpR.readlines()    
    for img_path in lines:
        img_path = img_path.strip()
        print(img_path)
        img = cv2.imread(img_path)
        begin_time = time.time()
        yolov5_obj.process(img, yolov5.IMG_BGR)
        rects = yolov5_obj.get_result()
        end_time = time.time()
        eps = (end_time - begin_time) * 1000
        print("Average time is %fms"%(eps))
        for ind, rect in enumerate(rects):
            xmin = int(rect.x1)
            ymin = int(rect.y1)
            xmax = int(rect.x2)
            ymax = int(rect.y2)
            print("Detected %d obj : "%(ind), (xmin, ymin), (xmax, ymax))
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 125, 100), 8, cv2.FONT_HERSHEY_PLAIN)
