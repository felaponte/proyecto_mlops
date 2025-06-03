import streamlit as st

# Importar las páginas (importa tu archivo de landing_page y prediction_page)
from landing_page import landing_page
from prediction import prediction_page

# Verificar si el estado de la página está en la sesión
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Navegar entre la landing page y la página de predicción
if st.session_state.page == 'landing':
    landing_page()

elif st.session_state.page == 'prediction':
    prediction_page()