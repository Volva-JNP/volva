# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 13:58:49 2022

@author: User
"""


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
from functions.volva_fct import *


path = 'datas/volva_datas_utlimate_one.csv'
path_brut = 'datas/volumesMARS2021.csv'






def set_data():
    df = load_csv(path)
    st.title('Data')
    st.write('select dataframe:')
    
    
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    menu = st.radio(
    "",
    ("secteur frais", "secteur Gel", "secteur FFL"),
)
    if menu =='secteur frais':
        df = df.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FFL'], axis = 1)
        secteur = 'REALISE_TOTAL_FRAIS'
        
    
    if menu == 'secteur Gel':
        df = df.drop(['REALISE_TOTAL_FRAIS','REALISE_TOTAL_FFL'], axis = 1)
        secteur = 'REALISE_TOTAL_GEL'
        
        
    if menu == 'secteur FFL':
        df = df.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FRAIS'], axis = 1)
        secteur = 'REALISE_TOTAL_FFL'
        
            
    
    st.write(df[['DATE',secteur]])
    
    st.write('<style>div.column-widget.stRadio > div{flex-direction:column;justify-content: center;} </style>', unsafe_allow_html=True)