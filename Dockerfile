FROM python:3.10

WORKDIR /app

ADD . /app

RUN apt update && apt install -y build-essential
RUN mkdir /app/uploaded_files
RUN mkdir /app/output_files

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/openai/whisper.git
RUN pip install -q git+https://github.com/pyannote/pyannote-audio > /dev/null
RUN apt install -y ffmpeg

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
