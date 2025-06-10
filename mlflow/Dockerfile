FROM python:3.9-slim

WORKDIR /mlflow

RUN pip install mlflow==2.8.1 psycopg2-binary==2.9.9

EXPOSE 5010

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5010"] 