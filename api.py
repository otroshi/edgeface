from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from verification import Verifier
import numpy as np
from io import BytesIO
from PIL import Image
import argparse
import uvicorn
import sys

app = FastAPI()
verifier = Verifier(model_name="edgeface_s_gamma_05 ")

def read_image_as_numpy(file: UploadFile) -> np.ndarray:
    try:
        image = Image.open(BytesIO(file.file.read())).convert("RGB")
        return np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1> Welcome to EdgeFace API </h1>"

@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    metric: str = "cosine"
):
    img1_np = read_image_as_numpy(image1)
    img2_np = read_image_as_numpy(image2)

    try:
        distance, threshold, is_same = verifier.verify(img1_np, img2_np, metric)
        return JSONResponse({
            "distance": float(distance),
            "threshold": float(threshold),
            "match": bool(is_same)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EdgeFace FastAPI server.")
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port number to run the server on (default: 8000)"
    )
    args = parser.parse_args()

    module_name = sys.argv[0].replace(".py", "")
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=args.port, reload=True)