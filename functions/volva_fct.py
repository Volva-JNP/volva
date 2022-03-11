
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 06:41:01 2022

@author: User
"""



'''fonction visualisation'''

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools
from texts.volva_text import *
from plotly.subplots import make_subplots



path = 'datas/volva_datas_utlimate_one.csv'
path_brut = 'datas/volumesMARS2021.csv'



#@st.cache
def load_csv(path):
    data = pd.read_csv(path, sep=',')
    return data








def set_home():
    st.image('img/siteSQF.png')
    with st.expander('Objectif'):
        st.write(intro, unsafe_allow_html=True)
    with st.expander('Définition et Origine'):
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(definition, unsafe_allow_html=True)
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.image('img/volvaeyesv1.gif')
    
    
    
    














#def set_model():
    #st.write('model')



#def set_pred():
 #   st.write('prediction')