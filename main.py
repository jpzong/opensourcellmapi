from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
import shutil
from typing import List
import subprocess
from osllm import OpenSourceLLM
import nest_asyncio
import asyncio
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": 'MODELOS_CARGADOS = ["llama2": "Llama2 (70B)","mixtral": "Mixtral (35B)","bakllava": "Bakllava (15B)"]'}

@app.post("/download_model/")
async def download_model(model: str = Query(..., title="Elección de modelo", description="Elección de modelo a descargar desde el repositorio de Ollama")):
    try:
        subprocess.run(['ollama', 'pull', model])
        return JSONResponse(content=jsonable_encoder({"message": f" Descargado el modelo {model} exitosamente"}), status_code=200)
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=500)

@app.get("/text-to-text/")
async def txt2txt(model: str = Query(..., title="Elección de modelo", description="Elección de modelo con opciones en: llama2, mixtral"),
                  prompt: str = Query(..., title="Prompt", description="Instrucción en la consulta al LLM")):
    try:
        llm = OpenSourceLLM(model)
        res = llm.text2text(prompt)
        return JSONResponse(content=jsonable_encoder({"message": f"{res}"}), status_code=200)
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=500)

@app.post("/image-to-text/")
async def image_to_text(file: UploadFile = File(...),prompt: str = Query(..., title="Image Prompt", description="Description or prompt for the image")):
    #self.rm_old_files()  # Clean up old files before processing
    try:
        with open(f"img/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_path = file.filename
        
        llm = OpenSourceLLM(model='bakllava')
        res = llm.img2text(file_path,prompt)
        return JSONResponse(content=jsonable_encoder({"message": f"{res}"}), status_code=200)
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=500)