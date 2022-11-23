from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image, ImageFilter
import import uvicorn

app = FastAPI()

@app.post("/")
def image_filter(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    original_image = original_image.filter(ImageFilter.BLUR)

    filtered_image = BytesIO()
    original_image.save(filtered_image, "JPEG")
    filtered_image.seek(0)

    return StreamingResponse(filtered_image, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("test_response_image:app", host="0.0.0.0", port=8000, log_level="info")
