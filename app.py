# Importando librerías
import streamlit as st
import pandas as pd
import numpy as np
import warnings

from PIL import Image

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

import xgboost as xgb

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Tech-Camp Oriente", page_icon=":house_buildings:", layout="wide")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>', unsafe_allow_html=True)

logo = Image.open('img/logo.png')
col1, col2 = st.columns([0.1, 0.9])

url = 'https://raw.githubusercontent.com/CJ7MO/price-predictor-system-ns/refs/heads/main/data/df_model.csv'
df = pd.read_csv(url)

reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, learning_rate=0.35)
X = df.drop(columns=['Precio'])
y = df['Precio']
reg.fit(X, y)

y_pred = reg.predict(X)
residuals = y - y_pred

residuals_df = pd.DataFrame({
    'Precios Predichos': y_pred if isinstance(y_pred, np.ndarray) else y_pred.values,  # Si ya es ndarray, lo usas directamente
    'Residuos': residuals if isinstance(residuals, np.ndarray) else residuals.values,  # Lo mismo para los residuos
    'Tipo': ['Positivo' if r > 0 else 'Negativo' for r in residuals]  # Clasificar en positivos o negativos
})


with col1:
    st.image(logo, width=250)

html_title = """
<style>
    .title-test {
    font-weight:bold;
    padding:5px;
    border-radius:6px
    }
</style>
<center><h1 class="title-test">Predicción del precio de Inmuebles en Norte de Santander</h1></center>
"""

with col2:    
    st.markdown(html_title, unsafe_allow_html=True)

st.subheader("Está aplicación web predice el precio de los inmuebles en Norte de Santander")

col3, col4 = st.columns([0.6, 0.4])

with col3:
    
    fig1 = px.scatter_3d(df, 
                        x='Habitaciones', 
                        y='Baños', 
                        z='Area', 
                        color='Precio',  
                        size='Precio',
                        color_continuous_scale=px.colors.sequential.YlOrBr, 
                        title='Grafico de las variables predictoras',
                        )

    fig1.update_layout(
        template='plotly_dark',
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)  
        ),
        width=800,   
        height=600,
        showlegend=False,  
    )
    
    st.plotly_chart(fig1)
    
with col4:

    fig2 = px.imshow(df.corr(), text_auto=True
                    ,color_continuous_scale=px.colors.sequential.YlOrBr, 
                    title='Correlación entre las variables')
    
    fig2.update_layout(
        width=800,  
        height=500,
    )
    
    st.plotly_chart(fig2)

col5, col6 = st.columns([0.6, 0.4])

with col5:
    
    fig3 = px.scatter(residuals_df, 
                    x='Precios Predichos', 
                    y='Residuos', 
                    color='Tipo',  # Diferenciar por color
                    title='Residuos vs Predicción',
                    labels={'Precios Predichos': 'Precios Predichos', 'Residuos': 'Residuos'},
                    color_continuous_scale=px.colors.sequential.YlOrBr,
                    )
    
    fig3.add_hline(y=0, line_dash="dash", line_color="red",
                )

    fig3.update_layout(
        width=900,   
        height=500,
        showlegend=False,  
        template='plotly_dark',
    )

    st.plotly_chart(fig3)

importance = reg.get_booster().get_fscore()

importance_df = pd.DataFrame({
    'Características': list(importance.keys()),
    'Puntuación F1': list(importance.values())
})

with col6:
    
    fig4 = px.bar(importance_df, 
                x='Puntuación F1', 
                y='Características',
                color='Características', 
                orientation='h',  
                title='Importancia de las características',
                color_continuous_scale=px.colors.sequential.YlOrBr,)

    fig4.update_layout(
        width=900,   
        height=500,
        showlegend=False, 
    )
    
    st.plotly_chart(fig4)
if 'precio_estimado' not in st.session_state:
    st.session_state.precio_estimado = ""

col7, col8, col9, col10 = st.columns([0.25, 0.25, 0.25, 0.25])

with col7:
    area_input = st.text_input("Área del Inmueble (m²):", placeholder="Ingresa el area del inmueble en m²")
    
with col8:
    habitaciones_input = st.text_input("Habitaciones del Inmueble:", placeholder="Ingresa las el número de habitaciones del inmueble")
    
with col9:
    baños_input = st.text_input("Baños del Inmueble:", placeholder="Ingresa el número de Baños del inmueble")

with col10:
    precio_input = st.text_input("Precio del Inmueble:", placeholder="Precio del inmueble según el modelo", value=st.session_state.precio_estimado)


st.markdown("""
    <style>
    .stButton > button {
        background-color: #FF4B4B;  /* Cambia este color a tu preferencia */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Convertir las entradas a float
# try:
#     area = float(area_input)
#     habitaciones = float(habitaciones_input)
#     baños = float(baños_input)
#     input_data = [[area, habitaciones, baños]]
# except ValueError:
#     st.warning("Por favor, ingresa valores numéricos en todos los campos.")

col1, col2, col3 = st.columns([0.4690, 0.062, 0.4690])
with col2:
    if st.button("Predecir", key="predict_button"):
        try:
            #precio_estimado = reg.predict(input_data)[0]
            if 'key' not in st.session_state:
                st.session_state['key'] = st.session_state.precio_estimado
            #st.session_state.precio_estimado = f"${precio_estimado:,.2f}"
            st.session_state.button_text = "Predicción realizada"
        
        except ValueError:
            st.warning("Por favor, ingresa valores válidos en todos los campos.")
        
        st.rerun()