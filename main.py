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

llm = OpenSourceLLM()
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

@app.get("/image-to-text/")
async def image_to_text(filename: str = Query(..., title="Nombre del archivo de imagen", description="Nombre del archivo de imagen"),
                          prompt: str = Query(..., title="Image Prompt", description="Description or prompt for the image")):
    """Procesa una imagen y genera texto utilizando un modelo de lenguaje."""

    self.rm_old_files()  # Clean up old files before processing

    file_path = f"img/{filename}"
    try:
        pil_image = Image.open(file_path)
        image_b64 = self.convert_to_base64(pil_image)
        generated_text = self.img2text(filename, prompt)

        response_content = {
            "message": "Image-to-text generation successful",
            "generated_text": generated_text
        }
        return response_content

    except Exception as e:
        return {"message": f"Error al procesar la imagen: {e}"}

nest_asyncio.apply()

config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, loop='asyncio')
server = uvicorn.Server(config)
await server.serve()