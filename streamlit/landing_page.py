import streamlit as st

def landing_page():
    # Usar st.image para poner una imagen como fondo de la página principal
    page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://images.ctfassets.net/gtoyaf5jfz2g/3T5ToOyk5EidSQdTOhddK6/9a84a58bab0cdd3bdb3ab7c68e857888/32105.jpg");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: local;
        }}
        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}
        </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Título y contenido de la landing page
    st.title('¡Bienvenido al Proyecto del grupo 02!')

    # Incrustar el video de YouTube
    st.markdown(
        """
        <div style="text-align: center;">
            <h3>¡Nuestro video del desarrollo del proyecto!</h3>
            <iframe width="560" height="315" src="https://www.youtube.com/embed/E7ns_VI3fGE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
        """, unsafe_allow_html=True
    )

    # Botón para ir a la interfaz de predicción
    if st.button('¡Empezar la predicción!'):
        st.session_state.page = 'prediction'  # Cambiar a la página de predicción
