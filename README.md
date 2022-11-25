## FastAPI deep learning object detect(yolov5) deploy

- 软件环境：

	- Ubuntu 20.04/Win10
	
	- Python 3.7.x, fastapi, python-multipart, uvicorn, opencv-python, virtualenv

- Demo

	- 服务端
```shell
# 推荐使用virtualenv单独设置Python环境
python fastapi_demo_det.py
INFO:     Started server process [422]
INFO:     Waiting for application startup.
Init success!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

	- 客户端

```shell
# 1.通过http工具httpie, Ubuntu下通过sudo apt install httpie安装
http --form POST http://10.78.9.128:8000/detection image@test_imgs/20210422_102007.jpg

# 2.浏览器访问，打开浏览器
http://10.78.9.128:8000/docs#/default/detection_detection_post

# 3.Python代码访问
python client_demo.py
```


## 其它

目前检测python库通过pybind11封装c++库实现

