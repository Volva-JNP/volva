# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:12:33 2022

@author: User
"""

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
import webbrowser


st.sidebar.image('./img/Les_Mousquetaires.png')
st.sidebar.write("")

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


st.sidebar.header('MENU')
st.sidebar.markdown('analyse de données ITM SQF')






button_intro = st.sidebar.button('Introduction')
button_data = st.sidebar.button('Data')
button_visu = st.sidebar.button('Visualisation')
button_model = st.sidebar.button('Modèlisation')
button_predict = st.sidebar.button('Prédictions')


if button_intro:
    set_home()  

if button_data:
    set_data() 

if button_visu:
    set_visu() 

if button_model:
    set_visu() 

if button_predict:
    set_visu() 








