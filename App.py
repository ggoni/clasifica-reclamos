import pandas as pd
import numpy as np
import streamlit as st
import pickle


st.title('Clasificador de Feedback de Clientes')

st.markdown(
    '## Viene de la pregunta: _Cuéntanos, ¿por qué calificas con esa nota la experiencia en X?_')

st.subheader('1) Clasificar sólo Un comentario')

comentario_input = st.text_area('Comentario:')


with open("modelo_vectorizador.pckl", 'rb') as archivo_in:
    vectorizador = pickle.load(archivo_in)

with open("modelo_clasificador.pckl", 'rb') as archivo_in:
    clasificador = pickle.load(archivo_in)


def reemplaza_caracter_erroneo(palabra):
    palabra = palabra.replace('&#225;', 'á')
    palabra = palabra.replace('&#193;', 'á')
    palabra = palabra.replace('&#233;', 'é')
    palabra = palabra.replace('&#201;', 'é')
    palabra = palabra.replace('&#237;', 'í')
    palabra = palabra.replace('&#205;', 'í')
    palabra = palabra.replace('&#243;', 'ó')
    palabra = palabra.replace('&#242;', 'ó')
    palabra = palabra.replace('&#211;', 'ó')
    palabra = palabra.replace('&#218;', 'ú')
    palabra = palabra.replace('&#249;', 'ú')
    palabra = palabra.replace('&#250;', 'ú')
    palabra = palabra.replace('&#241;', 'ñ')
    palabra = palabra.replace('&#209;', 'ñ')
    palabra = palabra.replace('&#252;', 'u')
    palabra = palabra.replace('&#176;', 'º')
    palabra = palabra.replace('caros', 'caro')
    return palabra


diccionario_inv = {0: 'Precio', 1: 'Producto',
                   2: 'Sala/vendedores', 3: 'Otros'}


def predice_frase(frase: str):
    frase = reemplaza_caracter_erroneo(frase).lower()
    mapeo = vectorizador.transform([frase])
    label = clasificador.predict(mapeo).astype(int)
    return diccionario_inv[int(label)]


y = ''

if st.button("Clasifica"):
    y = predice_frase(comentario_input)

else:
    st.write("En espera")

#y = predice_frase(comentario_input)

mensaje = 'Se clasifica como: '+y.upper()

st.markdown('**_' + mensaje+'_**')
