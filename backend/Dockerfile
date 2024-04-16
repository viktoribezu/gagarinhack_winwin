FROM python:3.10
RUN apt-get update \
    && apt-get install tesseract-ocr -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt