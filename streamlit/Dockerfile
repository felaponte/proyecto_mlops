# Usa una imagen base de Python
FROM python:3.9

WORKDIR /streamlit

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "./app.py", "--server.address=0.0.0.0", "--server.port=8501"]
