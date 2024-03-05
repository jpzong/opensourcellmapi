from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
import shutil
from typing import List
import subprocess
from osllm import OpenSourceLLM

app = FastAPI()

llm = OpenSourceLLM()
@app.get("/")
def read_root():
    return {"message": 'MODELOS_CARGADOS = ["llama2": "Llama2 (70B)","mixtral": "Mixtral (35B)","bakllava": "Bakllava (15B)"]'}

@app.pos("/download_model/")
async def download_model(model: str = Query(..., title="Elección de modelo", description="Elección de modelo a descargar desde el repositorio de Ollama")):
    try:
        subprocess.run(['ollama', 'pull', model])
        return JSONResponse(content=jsonable_encoder({"message": f" Descargado el modelo {model} exitosamente"}), status_code=200)
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=500)

@app.get("/txt2txt/")
async def txt2txt(model: str = Query(..., title="Elección de modelo", description="Elección de modelo con opciones en: llama2, mixtral"),
                  prompt: str = Query(..., title="Prompt", description="Instrucción en la consulta al LLM")):
    try:
        res = llm.text2text(prompt)
        return JSONResponse(content=jsonable_encoder({"message": f"{res}"}), status_code=200)
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=500)

@app.get("/speech-to-text/")
async def speech_to_text(speakers: str = Query(..., title="Speaker Name", description="Name of the speaker"),
                         language: str = Query(..., title="Language", description="Language for speech-to-text")):
    response_content = {
        "message": f"speech-to-text conversion for {speakers} in {language} language."
    }
    return response_content

@app.get("/get-transcript/")
async def get_text_file(name_file: str = Query(..., title="Nombre del archivo", description="Nombre del archivo")):
    file_path = f"/workspace/transcript.txt"
    transcription_processor.run_transcription('/workspace/' + name_file)
    return FileResponse(file_path, filename=f"transcript.txt", media_type="text/plain")