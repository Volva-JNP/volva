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



st.sidebar.image('img/Les_Mousquetaires.png')
st.sidebar.write("")

if 'page' not in st.session_state:
    st.session_state.page = 'Intro'



m = st.markdown("""

<style>
div.stButton >
    button:first-child {
        background-color:transparent;
        border-radius:6px;
        cursor:pointer;        
        font-family:Arial;
        font-size:15px;
        font-weight:bold;

        margin-right: 0px;
        height: 25px;
        width: 250px

    }
</style>""", unsafe_allow_html=True)


#menu selection du dataset
Data=False

st.sidebar.header('MENU')
st.sidebar.markdown('analyse de données ITM SQF')


button_intro = st.sidebar.button('Introduction')
button_data = st.sidebar.button('Data')

button_visu = st.sidebar.button('Visualisation')
if button_visu:
    link='[Moyenne Mobile 2020-21](#moyenne-mobile-volume-par-secteur)'
    st.sidebar.markdown(link,unsafe_allow_html= True )
    link2='[Distribution par secteur](#distribution-des-volumes-par-secteur)'
    st.sidebar.markdown(link2,unsafe_allow_html= True )
    link1='[volume moyen par jour](#volume-par-jour)'
    st.sidebar.markdown(link1,unsafe_allow_html= True )
    link3='[Jour Férié](#impact-jour-f-ri)'
    st.sidebar.markdown(link3,unsafe_allow_html= True )
button_model = st.sidebar.button('Modèlisation')
button_predict = st.sidebar.button('Prédictions')


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
    set_visu()


 




    
 






