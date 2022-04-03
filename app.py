# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:12:33 2022

@author: User
"""



from doctest import DocFileSuite
import streamlit as st
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from functions.volva_fct import *
from texts.volva_text import *
from functions.volva_fct_visu import *
from functions.volva_fct_data import *
from functions.volva_fct_model import *
from functions.volva_fct_predict import *
from functions.volva_fct_add_datas import *

st.set_page_config(page_title='Volva', page_icon='img/favicon/android-chrome-192x192.png')

with open('css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.markdown(f'<center>', unsafe_allow_html=True)

st.sidebar.image('img/volvaF1.png')
st.sidebar.image('img/projet_volva.png')


if 'page' not in st.session_state:
    st.session_state.page = 'Intro'



#menu selection du dataset
Data=False

st.sidebar.header('MENU')

button_intro = st.sidebar.button('Introduction')
# if button_intro:

button_data = st.sidebar.button('Data')
# if button_data:

button_visu = st.sidebar.button('Visualisation')
if button_visu:
    link='[Moyenne Mobile 2020-21](#moyenne-mobile-volume-par-secteur)'
    st.sidebar.markdown(link,unsafe_allow_html= True )
    link2='[Distribution par secteur](#distribution-des-volumes-par-secteur)'
    st.sidebar.markdown(link2,unsafe_allow_html= True )
    link1='[volume moyen par jour](#volume-moyen-jour-par-secteur)'
    st.sidebar.markdown(link1,unsafe_allow_html= True )
    link3='[Jour Férié](#impact-jour-f-ri)'
    st.sidebar.markdown(link3,unsafe_allow_html= True )

button_model = st.sidebar.button('Modélisation')
if button_model:
    link='[Données utiles par secteur](#selection-des-donn-es-utiles-par-test-de-mod-les)'
    st.sidebar.markdown(link,unsafe_allow_html= True )

    link='[Tests modèles de regression](#tests-des-mod-les-de-regression)'
    st.sidebar.markdown(link,unsafe_allow_html= True )

    link='[Comparaison des Modèles](#comparaison-des-mod-les)'
    st.sidebar.markdown(link,unsafe_allow_html= True )

    link='[Modèle final](#construction-du-mod-le-final)'
    st.sidebar.markdown(link,unsafe_allow_html= True )

button_predict = st.sidebar.button('Prévisions')
if button_predict:   
    link='[Prédictions sur période](#pr-dictions-sur-une-p-riode)'
    st.sidebar.markdown(link,unsafe_allow_html= True )

button_add_datas = st.sidebar.button('Ajouter des données')





if button_intro:
        st.session_state.page = 'Intro'
    

if button_data:
    st.session_state.page = 'Data'
    
    
if button_visu:
    st.session_state.page = 'Vizu'
    

if button_model:
    st.session_state.page = 'Model'
 
if button_predict:
    st.session_state.page = 'Predict' 

if button_add_datas:
    st.session_state.page = 'add_datas' 
    


page = st.session_state.page 

if page == 'Intro':
    set_home() 

if page == 'Data':
    set_data()

if page == 'Vizu':
    set_visu()

if page == 'Model':
    build_page_model()

if page == 'Predict':
    build_page_predict()

if page == 'add_datas':
     build_page_add_datas()



st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.markdown(f'<center>created by :</center>', unsafe_allow_html=True)
st.sidebar.write("")
st.sidebar.markdown(f'<center><u>Volva Team</u></center>', unsafe_allow_html=True)
st.sidebar.write("")
link='[Phil Arrive](https://www.linkedin.com/in/philippe-arrive-954765137/)'
st.sidebar.markdown(link,unsafe_allow_html= True )
st.sidebar.image("img/Image4.jpg",width=50)

link='[Nicolas Francois](https://www.linkedin.com/in/nicolas-francois-finance-and-bi/)'
st.sidebar.markdown(link,unsafe_allow_html= True )
st.sidebar.image("img/Image6.jpg",width=50)

link='[Julien Khenniche](https://www.linkedin.com/in/philippe-arrive-954765137/)'
st.sidebar.markdown(link,unsafe_allow_html= True )
st.sidebar.image('img/Image5.jpg',width=50)

st.sidebar.write("")
st.sidebar.markdown(f'<center>and</center>', unsafe_allow_html=True)
st.sidebar.markdown(f'<center>Paul Datascientest as the Mentor</center>', unsafe_allow_html=True)
    








