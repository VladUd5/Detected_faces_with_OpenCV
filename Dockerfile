FROM python:3.10
WORKDIR /Face_recognition

COPY requirements.txt /Face_recognition/requirements.txt
RUN pip install --upgrade pip
RUN apt update; apt install -y libgl1
RUN pip3 install -r requirements.txt
COPY . /Face_recognition

VOLUME /Face_recognition

RUN chmod +x /Face_recognition/face_recognition.py
CMD ["python3","/Face_recognition/face_recognition.py"]
