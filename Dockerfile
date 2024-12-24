FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app
COPY ./src /app
COPY download_model.py /app

RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

RUN python download_model.py model.h5

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
