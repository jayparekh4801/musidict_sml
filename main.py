from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from src.pipelines import predict_pipeline
from src.components.array_column_transformer import ArrayColumnTransformer
from src.utils import load_object
from fastapi.middleware.cors import CORSMiddleware

model = predict_pipeline.PredictPipeline()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Directory to store uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_audio(
    bit_rate: int = Form(...),
    genre: str = Form(...),
    duration: int = Form(...),
    audio_file: UploadFile = Form(...)
):
    file_path = os.path.join(UPLOAD_DIR, audio_file.filename)
    with open(file_path, "wb") as f:
        f.write(await audio_file.read())

    # Simulate saving metadata
    metadata = {
        "bit_rate": bit_rate,
        "genre": genre,
        "duration": duration,
        "file_path": file_path
    }
    print(metadata)
    result = model.predict(metadata)
    print(result[0][0])
    return JSONResponse(content={
        "message": f"Audio file converted to MP3 and uploaded successfully!",
        "data": result[0][0]
        })



