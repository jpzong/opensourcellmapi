from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.llms import OpenAI
from langchain_community.llms import Ollama
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import PyPDF2

import base64
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image
import os
import time

class OpenSourceLLM:
    def __init__(self, model="llama2"):
        self.llm = Ollama(
            model=model, #mixtral
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )        

    def rm_old_files(self):
        img_dir = "img"
        tiempo_actual = time.time()
        un_dia = 24 * 60 * 60
        os.chdir(img_dir)
        for filename in os.listdir():
            filepath = os.path.join(img_dir, filename)
            # Verificar si el archivo es más antiguo que un día
            if os.path.isfile(filepath) and tiempo_actual - os.path.getmtime(filepath) > un_dia:
                os.remove(filepath)

    def convert_to_base64(self,pil_image):
        """
        Convert PIL images to Base64 encoded strings

        :param pil_image: PIL image
        :return: Re-sized Base64 string
        """

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")  # You can change the format if needed
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def img2text(self,filename,prompt):
        file_path = "img/"+filename
        pil_image = Image.open(file_path)
        image_b64 = self.convert_to_base64(pil_image)
        llm_with_image_context = self.llm.bind(images=[image_b64])
        return llm_with_image_context.invoke(prompt)

    def text2text(self,prompt):
         return self.llm.invoke(prompt)

def text2chatbot(self, filename):
    try:
        if filename.endswith('.pdf'):
            with open(filename, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages)
        elif filename.endswith(".txt"):
            with open(filename, "r") as text_file:
                text = text_file.read()
        else:
            raise ValueError("Formato de archivo no soportado")

        relevant_text = text[:1000]

        prompt = f"Por favor proporcioname un resumen y aspectos clave acerca del siguiente texto:\n{relevant_text}"
        response = self.llm.invoke(prompt)
        print(response)

        while True:
            user_query = input("Escribe una pregunta acerca del texto: (o escribe 'cerrar' para terminar): ")
            if user_query.lower() == "cerrar":
                break
            response = self.llm.invoke(user_query, context=text)
            print(response)

    except Exception as e:
        print("Error procesando el archivo:", e)