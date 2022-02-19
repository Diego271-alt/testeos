from fastapi import FastAPI, UploadFile, File
from utils import process_image, create_model, animals

app = FastAPI()

model = create_model()
checkpoint_path = "training/cp.ckpt"
model.load_weights(checkpoint_path)

@app.post('/')
async def home(img: UploadFile = File(...)):  
    extension = img.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = process_image(await img.read())
    result = int(model.predict(image)[0][0] > 0.5)
    return {'resultado': animals[result]}