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
from diagrams import Cluster, Diagram
from diagrams.k8s.storage import PVC, PV
from diagrams.aws.database import Aurora
from diagrams.gcp.iot import IotCore
from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub
from diagrams.gcp.database import BigTable
from diagrams.gcp.compute import AppEngine, Functions
from diagrams.aws.database import Redshift, ElastiCache


path = 'datas/volva_datas_utlimate_one.csv'
path_brut = 'datas/volumesMARS2021.csv'






def set_data():
    df = load_csv(path)
    st.title('Data')
    
    
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
    
    st.write(DATA, unsafe_allow_html=True)
<<<<<<< HEAD
    flowchart()
    st.image("img/flowchart.jpg",width=1400)


def flowchart():
  with Diagram("Construction du dataset final", outformat="jpg", filename="img/flowchart"):
    
    with Cluster("DataSet Original"):
        with Cluster("Concatenation des mois"):
            dsoriginal_group = [PVC("mois 1"),
                                PVC("mois 2"),
                                PVC("mois 3"),
                                PVC("mois...")]
        dsconcat = BigTable("DataSet Concaténé")
        with Cluster("Nettoyage des données"):
            dscleaning = [IotCore("Sélection des variables"),
                          IotCore("Traitement des NA"),
                          IotCore("Traitement des outliers"),
                          IotCore("Recherche des données manquantes")]
        dsfullclean = Aurora("DataSet nettoyé")
        
        
    with Cluster("Intégration de données supplémentaires"):
        inthyp = Dataflow("Intégration des hypothèses")
#         with Cluster("Espace temps"):
#             flow >> Functions("La position du jour influe-t-elle sur les volumes ?") >> Redshift("Variables jours/semaines dans espace temps")
#         with Cluster("Jours fériés"):
#             flow >> Functions("La proximité d'un jour férié (passé ou à venir) influe-t-elle sur les volumes ?") >> Redshift("Variables prox jour férié")
#         with Cluster("Vacances scolaires"):
#             flow >> Functions("Les vacances scolaires (par zone) influecent-elles les volumes ?") >> Redshift("Variables vacances")
#         with Cluster("Températures moyennes saisonnières"):
#             flow >> Functions("Les températures des régions de livraison influencent-elles les volumes ?") >> Redshift("Variables températures")
#         with Cluster("Promotions"):
#             flow >> Functions("Les promotions influencent-elles les volumes ?") >> Redshift("Variables promotions")
#         with Cluster("Semaines spéciales"):
#             flow >> Functions("Les semaines spéciales influencent-elles les volumes ?") >> Redshift("Variables semaines spéciales")
        with Cluster("Espace temps"):
            esptemp = inthyp >> Functions() >> Redshift()
        with Cluster("Jours fériés"):
            jferies = inthyp >> Functions() >> Redshift()
        with Cluster("Vacances scolaires"):
            vac = inthyp >> Functions() >> Redshift()
        with Cluster("Températures moyennes saisonnières"):
            temp = inthyp >> Functions() >> Redshift()
        with Cluster("Promotions"):
            prom = inthyp >> Functions() >> Redshift()
        with Cluster("Semaines spéciales"):
            sem = inthyp >> Functions() >> Redshift()

        
    dsfinal = ElastiCache("DS final")
    
    dsanalyses = Redshift("Visualisation et analyses")
    
    dsoriginal_group >> dsconcat >> dscleaning >> dsfullclean >> inthyp 
    esptemp >> dsfinal
    jferies >> dsfinal
    vac >> dsfinal
    temp >> dsfinal
    prom >> dsfinal
    sem >> dsfinal
    
    
    dsfinal >> dsanalyses


    
=======

    st.image("img/flowchart.jpg",width=1400)



   
>>>>>>> 293f8c134fba5ea9772c4d917e26d4af9e522ec4
