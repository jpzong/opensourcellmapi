#!/bin/bash

# Crear directorio de imagenes
mkdir img
# Ejecutar el script install.sh desde ollama.ai
curl https://ollama.ai/install.sh | sh

# Iniciar ollama serve en segundo plano y redirigir la salida a /dev/null
ollama serve >/dev/null 2>&1 &
# Realizar una descarga desde llama2:70b
ollama pull llama2:70b
# Realizar una descarga desde mixtral
ollama pull mixtral
# Realizar una descarga desde bakllava
ollama pull bakllava

# Instalar los paquetes de Python langchain y langchain_experimental usando pip
pip install -r requirements.txt

# Comenzar servidor de API
uvicorn main:app --reload --host 0.0.0.0 --port 8000