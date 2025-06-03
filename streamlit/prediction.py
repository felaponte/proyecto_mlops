import streamlit as st
import requests
import pandas as pd
from sqlalchemy import create_engine

def get_validations_table():
    engine = create_engine("mysql+pymysql://user:password@db-clean-data:3308/db")
    query = "SELECT * FROM validations ORDER BY Fecha DESC"
    return pd.read_sql(query, engine)

def prediction_page():
    st.title('Predicción de precio de vivienda 🏠')

    # Inputs del usuario
    brokered_by = st.number_input('Brokered By', value=0)
    bed = st.number_input('Número de habitaciones (bed)', value=0)
    bath = st.number_input('Número de baños (bath)', value=0)
    acre_lot = st.number_input('Tamaño del terreno (acre_lot)', value=0)
    house_size = st.number_input('Tamaño de la casa (house_size)', value=0)

    if st.button('¡Predecir!'):
        payload = {
            "brokered_by": brokered_by,
            "bed": bed,
            "bath": bath,
            "acre_lot": acre_lot,
            "house_size": house_size
        }

        try:
            response = requests.post(
                "http://api-service:8989/predict",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            st.success(f"💰 Precio estimado: {result['El precio de la vivienda es']:.2f}")
            st.info(f"🏷️ Entrenado en batch: {result['Modelo entrenado en batch']}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error al conectarse con la API: {e}")
    

    st.markdown("---")

    if st.button("📋 Ver historial de validaciones"):
        st.subheader("Historial de validaciones y despliegues")
        try:
            df_validations = get_validations_table()
            st.dataframe(df_validations)
        except Exception as e:
            st.error(f"❌ Error al cargar la tabla de validaciones: {e}")
