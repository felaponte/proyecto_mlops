from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, MetaData
import pymysql
from datetime import datetime
import os
import requests
import json
import time
from joblib import parallel_backend

#------------------------------------Restart data pipeline--------------------------------------
def restart_data():
    #Borrar tablas de DB
    engine1 = create_engine("mysql+pymysql://user:password@10.43.101.177:3307/db")
    
    meta1 = MetaData()
    meta1.reflect(bind=engine1)
    meta1.drop_all(bind=engine1)
    print("¡Todas las tablas de raw han sido eliminadas!")
    
    engine2 = create_engine("mysql+pymysql://user:password@10.43.101.177:3308/db")
    
    meta2 = MetaData()
    meta2.reflect(bind=engine2)
    meta2.drop_all(bind=engine2)
    print("¡Todas las tablas de clean han sido eliminadas!")
    
    
    # Hacer la petición
    url = "http://10.43.101.108:80/restart_data_generation"
    params = {
        "group_number": 2,
        "day": "Wednesday"
    }
    headers = {"accept": "application/json"}
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        # Convertimos la respuesta en DataFrame directamente
        print("Se reinicia data")
    
    

#---------------------------------------------------------------------------------------

with DAG(
    dag_id='restart_data_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    description='Restart data pipeline for houses dataset',
) as dag:

    restart_task = PythonOperator(
        task_id='restart_data',
        python_callable=restart_data
    )
    
    restart_task

