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
from texts.volva_text import *
from func.test import *





st.sidebar.image('./img/Les_Mousquetaires.png')
st.sidebar.write("")




#menu selection du dataset


st.sidebar.header('MENU')
st.sidebar.markdown('analyse de données ITM SQF')


menu = st.sidebar.radio(
    "",
    ("Intro", "Data", "visualisation", 'model', "prediction"),
)

if menu == 'Intro':
    set_home()  
elif menu == 'Data':
    set_data()
elif menu == 'visualisation':
    set_visu()
elif menu == 'model':
    set_model()
elif menu == 'prediction':
    set_pred()






