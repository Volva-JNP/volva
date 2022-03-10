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
# from diagrams import Cluster, Diagram
# from diagrams.k8s.storage import PVC, PV
# from diagrams.aws.database import Aurora
# from diagrams.gcp.iot import IotCore
# from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub
# from diagrams.gcp.database import BigTable
# from diagrams.gcp.compute import AppEngine, Functions
# from diagrams.aws.database import Redshift, ElastiCache


path = 'datas/volva_datas_utlimate_one.csv'
path_brut = 'datas/volumesMARS2021.csv'






def set_data():
    df = load_csv(path)
    st.title('Data')
    # with st.expander('Informations sur la construction du DF'):
        
    #     col3 = st.columns(1)
    #     with col3:
    #         st.write(data, unsafe_allow_html=True)
    
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Exemple d'un fichier mensuel")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.image('img/dataori.jpg')
            
            
    with col2:
        st.write('select dataframe:')
            
            
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            
        menu = st.radio(
        "",
        ("vide","secteur frais", "secteur Gel", "secteur FFL"),
        )
        if menu =='secteur frais':
            df = df.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FFL'], axis = 1)
            secteur = 'REALISE_TOTAL_FRAIS'
            st.write(df[['DATE',secteur]])    
            
        if menu == 'secteur Gel':
            df = df.drop(['REALISE_TOTAL_FRAIS','REALISE_TOTAL_FFL'], axis = 1)
            secteur = 'REALISE_TOTAL_GEL'
            st.write(df[['DATE',secteur]])    
                
        if menu == 'secteur FFL':
            df = df.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FRAIS'], axis = 1)
            secteur = 'REALISE_TOTAL_FFL'
    
            st.write(df[['DATE',secteur]])
    
    st.write('<style>div.column-widget.stRadio > div{flex-direction:column;justify-content: center;} </style>', unsafe_allow_html=True)
    
    
    #explication des données
    
    st.write("Explications des données et rapport d'exploration")
    
    st.write(data, unsafe_allow_html=True)

    st.image('img/construction_du_dataset_final.png',width=800)



   
