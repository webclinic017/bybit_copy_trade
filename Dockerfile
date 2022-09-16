FROM python:3.9

ENV TZ=Asia/Hong_Kong

ADD main.py .

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

COPY . .

CMD ["python3", "-u", "./main.py"]