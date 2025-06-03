<<<<<<< HEAD

=======
# Proyecto 3 - MLOps PUJ

Este proyecto tiene como objetivo la migración de Docker a Kubernetes para gestionar los servicios originalmente orquestados con Docker Compose. Para ello se utilizó la herramienta kompose convert, permitiendo desplegar múltiples servicios en un entorno Kubernetes completo.

Los servicios incluyen:

- Bases de datos MySQL (Raw y Clean)
- Jupyter para exploración y entrenamiento
- MLflow para registro de experimentos
- MinIO como sistema de almacenamiento para modelos
- Airflow para automatizar procesos como carga, limpieza y entrenamiento
- FastAPI como backend para inferencia
- Streamlit como interfaz de usuario
- Locust para pruebas de carga
- Prometheus para monitoreo
- Grafana para visualización de métricas

Desplegado en la máquina con IP 10.43.101.177

## Estructura del Proyecto

```
/9-Kubernetes
│
│── docker-compose.yaml               # Despliegue local de todos los servicios principales con Docker Compose
│── docker-compose.airflow.yaml      # Despliegue específico del servicio Airflow con Docker Compose
│── docker-compose-pvc.yaml          # Compose usado para migrar a Kubernetes solo los servicios con volumenes persistentes (MySQL)
│── docker-compose-hostPath.yaml     # Compose usado para migrar el resto de servicios sin PVC con Kompose
│── README.md                         # Documentación del proyecto
│
│── API/                              # Carpeta del servicio de inferencia vía API (FastAPI)
│   ├── api.py                        # Lógica principal de la API con FastAPI
│   ├── requirements.txt              # Dependencias necesarias para el servicio de API
│   └── Dockerfile                    # Imagen Docker de la API para subir a Docker Hub
│
│── jupyter-nb/                       # Carpeta del entorno de exploración y experimentación con Jupyter
│   ├── Dockerfile                    # Imagen Docker para Jupyter con MLflow
│   └── files/
│       ├── mlflow_notebook.ipynb     # Notebook para exploración y entrenamiento del modelo
│       ├── .ipynb_checkpoints/       # Checkpoints generados automáticamente por Jupyter
│       └── data/                     # Datos de entrada (CSV de diabetes)
│
│── Komposefiles/                     # Archivos YAML generados con `kompose convert` para despliegue con Kubernetes
│   ├── api-service-deployment.yaml
│   ├── api-service-service.yaml
│   ├── db-clean-data-deployment.yaml
│   ├── db-clean-data-service.yaml
│   ├── db-metadata-mlflow-deployment.yaml
│   ├── db-metadata-mlflow-service.yaml
│   ├── db-raw-data-deployment.yaml
│   ├── db-raw-data-service.yaml
│   ├── grafana-deployment.yaml
│   ├── grafana-service.yaml
│   ├── locust-deployment.yaml
│   ├── locust-service.yaml
│   ├── minio-deployment.yaml
│   ├── minio-service.yaml
│   ├── mlflow-service-deployment.yaml
│   ├── mlflow-service-service.yaml
│   ├── ml-service-deployment.yaml
│   ├── ml-service-service.yaml
│   ├── my-db-clean-persistentvolumeclaim.yaml   # PVC para base de datos "clean"
│   ├── my-db-raw-persistentvolumeclaim.yaml     # PVC para base de datos "raw"
│   ├── mysql-db-persistentvolumeclaim.yaml      # PVC para metadata de MLflow
│   ├── prometheus-deployment.yaml
│   ├── prometheus-service.yaml
│   ├── streamlit-app-deployment.yaml
│   └── streamlit-app-service.yaml
│
│── locust/                           # Carpeta del servicio de pruebas de carga con Locust
│   ├── locust.py                     # Script con decoradores para realizar las peticiones a la API
│   ├── requirements-locust.txt       # Dependencias de Python para Locust
│   └── Dockerfile.locust             # Imagen Docker para Locust
│
│── minio/                            # Carpeta con estructura de almacenamiento de MinIO
│   ├── .minio.sys/                   # Archivos internos del sistema de MinIO
│   └── mlflows3/                     # Bucket de artefactos para modelos registrados en MLflow
│
│── mlflow/                           # Configuración del servidor de MLflow
│   ├── Dockerfile                    # Imagen Docker de MLflow
│   ├── requirements.txt              # Dependencias del servidor MLflow
│   ├── script.sh                     # Script de arranque para MLflow
│   └── metadata/                     # Carpeta para los datos persistentes de la base de datos de MLflow
│
│── prometheus/                       # Configuración del servicio Prometheus
│   └── prometheus.yml                # Archivo de scraping de métricas y definición de volumen
│
│── Streamlit/                        # Servicio de visualización e interacción con el modelo vía Streamlit
│   ├── Dockerfile                    # Imagen Docker para Streamlit
│   ├── requirements.txt              # Dependencias de la app Streamlit
│   ├── landing_page.py               # Página de inicio de la aplicación web
│   ├── prediction.py                 # Script para realizar inferencias conectando a la API
│   └── app.py                        # Aplicación principal de Streamlit



```
## Tecnologías Utilizadas

* Python 3.9
* FastAPI
* Scikit-learn
* MLflow
* Joblib
* Pandas y NumPy
* Mysql
* Docker
* Docker Compose
* MinIO
* Locust
* UV
* Streamlit
* Kubernetes
* Prometheus
* Grafana

---
## Consideraciones
- Para despliegue local con Docker, se deben usar:
    - docker-compose.yaml y docker-compose.airflow.yaml.

- Antes de migrar a Kubernetes, las imágenes de Streamlit, Locust, FastAPI, MLflow y Jupyter deben estar cargadas en DockerHub (en s4g0/proyecto_kubernetes). Otros servicios usan imágenes oficiales.

- La migración con Kompose debe dividirse:
    - docker-compose-pvc.yaml → servicios con volumen persistente (MySQL)
    - docker-compose-hostPath.yaml → resto de servicios

- Ajustar puertos en los archivos Service generados por kompose convert.

- Modificar manualmente los PersistentVolumeClaim de MySQL a 1G.

- Asegurarse de usar - (guion medio) en nombres de servicios para compatibilidad con Kompose. Evitar _ (guión bajo), ya que puede generar errores.


## Video para desarrollo del proyecto

[text](https://youtu.be/9bdy6M_IM0Y)


