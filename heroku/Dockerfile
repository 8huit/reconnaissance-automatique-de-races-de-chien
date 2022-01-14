FROM python:3.8

WORKDIR /app

COPY . /app

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg' 'libsm6' 'libxext6'  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "server.py"]
