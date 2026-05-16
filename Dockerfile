FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]