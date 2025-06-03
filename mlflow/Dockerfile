# Usa una imagen base de Python
FROM python:3.9

# Definir el directorio de trabajo
WORKDIR /mlflow

# Copiar un archivo de prueba
COPY . /mlflow/
RUN chmod +x /mlflow/script.sh

# Instala los paquetes
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Comando por defecto al iniciar el contenedor
CMD ["ls", "-l", "/mlflow"]
