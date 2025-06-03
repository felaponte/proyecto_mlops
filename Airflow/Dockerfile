# Usa una imagen base de Python
FROM apache/airflow:2.6.0-python3.9
#-----------
ARG MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
ARG AWS_ACCESS_KEY_ID=admin
ARG AWS_SECRET_ACCESS_KEY=supersecret
#-----------
# Copia el archivo de requirements (dentro del contexto del usuario airflow)
COPY requirements.txt /tmp/requirements.txt

# Instala los paquetes
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["ls", "-l"]

