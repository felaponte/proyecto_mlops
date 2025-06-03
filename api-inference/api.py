# Configuración de entorno para acceso a MinIO
import os
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'


from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from threading import Lock
from sqlalchemy import create_engine

# Conectarse a MLflow y cargar el modelo en producción
mlflow.set_tracking_uri("http://mlflow-service:5000")

# Reemplaza este nombre por el nombre real del modelo registrado
MODEL_NAME = "Regresion_Lineal_modelo_produccion"

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/production")
    print(f"✔ Modelo '{MODEL_NAME}' cargado correctamente.")
except Exception as e:
    raise RuntimeError(f"❌ Error al cargar el modelo '{MODEL_NAME}': {str(e)}")


# Get the run_id from the model version in Production
client = MlflowClient()
prod_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
run = client.get_run(prod_version.run_id)
modelo_entrenado_en_batch = run.data.metrics.get("modelo_entrenado_en_batch")

# ───── FastAPI app ─────
app = FastAPI()

# Métricas Prometheus
REQUEST_COUNT = Counter('predict_requests_total', 'Total de peticiones de predicción')
REQUEST_LATENCY = Histogram('predict_latency_seconds', 'Tiempo de latencia de predicción')

# Definir la estructura de los datos de entrada
class InputData(BaseModel):
    brokered_by: float
    bed: float
    bath: float
    acre_lot: float
    house_size: float

@app.get("/")
def home():
    return {"message": "¡API de predicción de precio de Inmueble!"}


# Lock to ensure thread safety when reloading the model
model_lock = Lock()

@app.post("/reload-model")
def reload_model():
    global model
    global modelo_entrenado_en_batch
    with model_lock:
        try:
            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/production")
            prod_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
            run = client.get_run(prod_version.run_id)
            modelo_entrenado_en_batch = run.data.metrics.get("modelo_entrenado_en_batch")
            return {"message": f"✔ Modelo '{MODEL_NAME}' recargado exitosamente."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ Error al recargar el modelo: {str(e)}")

@app.post("/predict")
async def predict(data: InputData):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        df = pd.DataFrame([data.dict()])
        with model_lock:
            pred = await run_in_threadpool(model.predict, df)

        # Add prediction and metadata to the dataframe
        df["predicted_price"] = pred[0]
        df["modelo_entrenado_en_batch"] = modelo_entrenado_en_batch

        # Connect to MySQL and append the row
        try:
            engine = create_engine("mysql+pymysql://user:password@db-raw-data:3307/db")
            df.to_sql("data_predictions", con=engine, index=False, if_exists="append")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ Error al guardar la predicción en la base de datos: {str(e)}")

    return {
        "El precio de la vivienda es": pred[0],
        "Modelo entrenado en batch": modelo_entrenado_en_batch
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
