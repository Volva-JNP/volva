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
from volva_text import *



#@st.cache
def load_csv(path):
    data = pd.read_csv(path, sep=',')
    return data





def violon(dataset,x,y):
    fig = px.violin(dataset,x= x, y=x, color= x, box = True)
    return fig
    


def set_home():
    st.image('itmsqf.jpg')
    st.write(intro, unsafe_allow_html=True)
    
    
    
    
def set_data():
    df = load_csv(path)
    st.title('Data')
    st.write('select dataframe:')
    
    col1,col2 = st.columns(2)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    with col1:
        if st.checkbox('secteur frais'):
            df = df.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FFL'], axis = 1)
            
    
        if st.checkbox('secteur gel'):
            df = df.drop(['REALISE_TOTAL_FRAIS','REALISE_TOTAL_FFL'], axis = 1)
            
        
        if st.checkbox('secteur ffl'):
            df = df.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FRAIS'], axis = 1)
            
    with col2:
        st.write(df)


def set_visu():
    dataset = load_csv(path)
    x='JOUR'
    if st.checkbox('secteur frais'):
        dataset = dataset.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FFL'], axis = 1)
        y='REALISE_TOTAL_FRAIS'
        fig = px.violin(dataset,x= x, y=y, color= x, box = True)
        st.write(fig)
    
    
    if st.checkbox('secteur gel'):
        dataset = dataset.drop(['REALISE_TOTAL_FRAIS','REALISE_TOTAL_FFL'], axis = 1)
        y='REALISE_TOTAL_GEL'
        fig = px.violin(dataset,x= x, y=y, color= x, box = True)
        st.write(fig)
        
    if st.checkbox('secteur ffl'):
        dataset = dataset.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FRAIS'], axis = 1)
        y='REALISE_TOTAL_FFL'
        fig = px.violin(dataset,x= x, y=y, color= x, box = True)
        st.write(fig)

    

def set_model():
    st.write('model')



def set_pred():
    st.write('prediction')
