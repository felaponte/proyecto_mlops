from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine
import pymysql
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import requests
import json
import numpy as np
import time
from joblib import parallel_backend
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler

#------------------------------------Load raw data--------------------------------------
def load_data():
    # Hacer la petición
    url = "http://10.43.101.108:80/data"
    params = {"group_number": 2, "day": "Wednesday"}
    headers = {"accept": "application/json"}
    
    response = requests.get(url, params=params, headers=headers)
    engine = create_engine("mysql+pymysql://user:password@10.43.101.177:3307/db")
    
    if response.status_code == 200:
        # Convertimos la respuesta en DataFrame directamente
        df_raw = pd.DataFrame(response.json())
        # Normalizamos la columna 'data'
        df_data = pd.json_normalize(df_raw["data"])
        # Combinamos con las columnas del DataFrame original (excepto 'data')
        df = pd.concat([df_raw.drop(columns=["data"]), df_data], axis=1)
        #Remover duplicados
        df.drop_duplicates(inplace=True)
        # Inserta el DataFrame como una tabla (esto reemplaza si ya existe)
        df.to_sql("data_raw", con=engine, index=False, if_exists="append")
    else:
        print(f"Error en la petición: {response.status_code}")
        print(response.text)
    
    
#------------------------------------Load transformed data--------------------------------------
def preprocesamiento():
    engine1 = create_engine("mysql+pymysql://user:password@10.43.101.177:3307/db")
    df = pd.read_sql_table("data_raw", engine1)
    
    df.drop(columns=['group_number','day','status','street','city','state','zip_code','prev_sold_date'], inplace=True)
    df = df[df['batch_number'] == df['batch_number'].max()].dropna()
        
    # Selección de variables predictoras y objetivo
    X = df.drop(columns=['price'])
    y = df['price']
    
    
    # Dividir en entrenamiento y prueba (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Guardar los conjuntos como tablas SQL (reemplazando si existen)
    engine2 = create_engine("mysql+pymysql://user:password@10.43.101.177:3308/db")
    X_train.to_sql("train_data_X", con=engine2, if_exists="append", index=False)
    X_test.to_sql("test_data_X", con=engine2, if_exists="append", index=False)
    y_train.to_sql("train_data_y", con=engine2, if_exists="append", index=False)
    y_test.to_sql("test_data_y", con=engine2, if_exists="append", index=False)
    
    print("Preprocesamiento completo. Tablas 'train_data' y 'test_data' creadas en MySQL.")
    
    
#------------------------------------Training--------------------------------------

def training_modelos_y_validacion():
    engine = create_engine("mysql+pymysql://user:password@10.43.101.177:3308/db")
    
    # Separar variables predictoras y objetivo
    X_train = pd.read_sql_table("train_data_X", engine)
    y_train = pd.read_sql_table("train_data_y", engine).values.ravel()
    X_test = pd.read_sql_table("test_data_X", engine)
    y_test = pd.read_sql_table("test_data_y", engine).values.ravel()
    
    max_batch = int(X_train['batch_number'].max())
    
    Validations = pd.DataFrame(columns=["batch_number", "Validaciones-hechas", "Despliegue_de_nuevo_modelo", "Fecha"])
    
    training_flag = False
    shift_detected = False
    
    if max_batch==0:
        text_validaciones = "No se hacen validaciones ya que son los primeros datos. Se entrena el primer modelo de todos."
        training_flag = True
    else:
        X_train_last = X_train[X_train['batch_number'] == max_batch].drop(columns=['batch_number'])    # filas con batch máximo
        X_train_rest = X_train[X_train['batch_number'] != max_batch].drop(columns=['batch_number'])    # filas con el resto del dataframe
        
        ratio = len(X_train_last) / len(X_train_rest)
        
        scaler = StandardScaler()
        X_train_rest_scaled = pd.DataFrame(scaler.fit_transform(X_train_rest), columns=X_train_rest.columns)
        X_train_last_scaled = pd.DataFrame(scaler.transform(X_train_last), columns=X_train_last.columns)
    
        
        if ratio <0.05:
            text_validaciones = f"Los nuevos datos representan menos del 5% de la data total, exactamente un {ratio*100}%. No se hacen más validaciones. Ni se entrena nuevo modelo."
            test_despliegue = "No hay nuevo modelo para desplegar"
        else:
            variables_con_drift = []
            if len(X_train_last) > len(X_train_rest):
                X_train_last=X_train_last.sample(len(X_train_rest), random_state=42)
            else:
                X_train_rest=X_train_rest.sample(len(X_train_last), random_state=42)
                
                
            for col in X_train_rest.columns:
                #stat, p_value  = ks_2samp(X_train_last[col], X_train_rest[col])
                #Print(KS)
                #if p_value < 0.05:
                #    shift_detected = True
                #    training_flag = True
                #    variables_con_drift.append(f"Se detecta data drift en la variable '{col}' con un p-value de {p_value:.4f}.")
                dist = wasserstein_distance(X_train_rest_scaled[col], X_train_last_scaled[col])
                print(dist)
                if dist > 0.1:  # define tu umbral empírico
                    print(f"Data drift en '{col}' con distancia Wasserstein = {dist:.4f}")
                    shift_detected = True
                    training_flag = True                
                    variables_con_drift.append(f"Se detecta data drift en la variable '{col}' con una distancia de Wasserstein de {dist:.4f}.")
                  
                    # Plot KDE
                    #import matplotlib.pyplot as plt
                    #import seaborn as sns
                    #plt.figure(figsize=(8, 4))
                    #sns.kdeplot(X_train_rest[col], label="Históricos", fill=True)
                    #sns.kdeplot(X_train_last[col], label="Último batch", fill=True)
                    #plt.title(f"Drift detectado en '{col}' (p = {dist:.4f})")
                    #plt.xlabel(col)
                    #plt.ylabel("Densidad")
                    #plt.legend()
                    #plt.grid(True)
                    #plt.tight_layout()
                    #plt.show()
                    
            if shift_detected:
                texto_largo = " ".join(variables_con_drift)
                text_validaciones = f"Existe datadrift en los datos. {texto_largo}. Se entrena nuevo modelo."
            else:
                text_validaciones = "No existe datadrift en los datos. No se entrena nuevo modelo."
                test_despliegue = "No hay nuevo modelo para desplegar"
                
    if training_flag:
        #----------------------------------------Variables de entorno MLFlow--------------------------------
        # connects to the Mlflow tracking server that you started above
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.101.177:9000"
        os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
        mlflow.set_tracking_uri("http://10.43.101.177:5000")
          
        #---------------------------------------------------------------------------
        mlflow.set_experiment("mlflow_tracking_houses_prices")
        mlflow.autolog(log_model_signatures=True, log_input_examples=True)
        
        # run description (just metadata)
        desc = "the simplest possible example"
        
        # executes the run
        with mlflow.start_run(run_name="Regresion_Lineal", description="Modelo de regresión lineal") as run:
            # Entrenar RandomForest
            # Modelo
            lr_model = LinearRegression()
            lr_model.fit(X_train.drop(columns=['batch_number']), y_train)
            # Predicción
            y_pred = lr_model.predict(X_test.drop(columns=['batch_number']))
            # Calcular accuracy
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mlflow.log_metric("rmse_test", rmse)
            mlflow.log_metric("modelo_entrenado_en_batch", max_batch)
            
        client = MlflowClient()
        experiment = client.get_experiment_by_name("mlflow_tracking_houses_prices")
        experiment_id = experiment.experiment_id
        
        # Obtener el modelo registrado más reciente
        model_name = "Regresion_Lineal_modelo_produccion"
        current_run_id = run.info.run_id
        
        if max_batch==0:
            # No hay modelo en producción → registrar y poner este
            model_uri = f"runs:/{current_run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True
            )
            test_despliegue = f"Se despliega primer modelo con RMSE igual a {rmse:.4f}."
        
        else:
            # 1. Obtener la versión actual en producción
            production_versions = client.get_latest_versions(model_name, stages=["Production"])
            prod_run_id = production_versions[0].run_id
            #prod_metrics = client.get_run(prod_run_id).data.metrics
            #rmse_prod = prod_metrics.get("rmse_test")
            prod_model_uri = f"models:/{model_name}/Production" #Cargar el modelo en producción usando su URI
            prod_model = mlflow.pyfunc.load_model(prod_model_uri)
            y_pred_prod = prod_model.predict(X_test)#Usar el modelo en producción para predecir el nuevo test set
            rmse_prod = np.sqrt(mean_squared_error(y_test, y_pred_prod))#Calcular RMSE con los nuevos datos
        
            if rmse < rmse_prod:
                # El nuevo es mejor → registrar y promover
                model_uri = f"runs:/{current_run_id}/model"
                mv = mlflow.register_model(model_uri, model_name)
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                test_despliegue = f"Se registró y desplegó nuevo modelo con RMSE {rmse:.4f}, mejor que el anterior evaluado sobre el nuevo test ({rmse_prod:.4f})."
            else:
                test_despliegue = f"El nuevo modelo tiene RMSE {rmse:.4f}, peor que el modelo en producción evaluado con el nuevo test ({rmse_prod:.4f}). No se despliega."
    
            
    nueva_fila = pd.DataFrame([{"batch_number": max_batch, "Validaciones-hechas": text_validaciones, "Despliegue_de_nuevo_modelo": test_despliegue, "Fecha": datetime.now()}])
    Validations = pd.concat([Validations, nueva_fila], ignore_index=True)
    Validations.to_sql("validations", con=engine, index=False, if_exists="append")


def reload_model():
    # URL de tu API (ajusta según puerto y host donde corre FastAPI)
    url = "http://10.43.101.177:8989/reload-model"
    
    # Realiza la petición POST
    response = requests.post(url)
    
    # Verifica el resultado
    if response.status_code == 200:
        print("✅ Recarga exitosa:", response.json())
    else:
        print("❌ Error al recargar el modelo:", response.status_code, response.text)


#---------------------------------------------------------------------------------------

with DAG(
    dag_id='Pipeline-proyecto-final',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    description='Load, preprocess, validate, and train models on price houses dataset',
) as dag:

    load_task = PythonOperator(
        task_id='load_raw_data',
        python_callable=load_data
    )
    
    load__transformed_task = PythonOperator(
        task_id='load_transformed_data',
        python_callable=preprocesamiento
    )
    
    training = PythonOperator(
        task_id='training_modelos_y_validacion',
        python_callable=training_modelos_y_validacion
    )
    
    reload_model = PythonOperator(
        task_id='reload_model',
        python_callable=reload_model
    )

    
    load_task >> load__transformed_task >> training >> reload_model

