# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 13:48:44 2022

@author: User
"""
import seaborn as sns
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
from functions.volva_fct import *
from PIL import Image

path = 'datas/volva_datas_utlimate_one.csv'
path_brut = 'datas/volva_datas.csv'



def set_visu():

    
    
        
    
    dataset = load_csv(path)
    df = load_csv(path)
  
    x='JOUR'
    df = df[(df['REALISE_TOTAL_FRAIS']>50000)&(df['REALISE_TOTAL_FRAIS']<160000)]
    dataset = df[(df['REALISE_TOTAL_FRAIS']>50000)&(df['REALISE_TOTAL_FRAIS']<160000)]
    df['total']= df['REALISE_TOTAL_FRAIS']+df['REALISE_TOTAL_FFL']+df['REALISE_TOTAL_GEL']
    df['moy_mob_total'] = df['total'].rolling(20).mean()

    df['month_day'] =  df['DATE'].apply(lambda date : date.split("-")[1] + date.split("-")[2])


    nb_jour_moy_mob = 20
    df['moy_mob_total'] = df['total'].rolling(nb_jour_moy_mob).mean()
    df['moy_mob_FRAIS'] = df['REALISE_TOTAL_FRAIS'].rolling(nb_jour_moy_mob).mean()
    df['moy_mob_FFL'] = df['REALISE_TOTAL_FFL'].rolling(nb_jour_moy_mob).mean()
    df['moy_mob_GEL'] = df['REALISE_TOTAL_GEL'].rolling(nb_jour_moy_mob).mean()
    df2020 = df[df['ANNEE']== 0][['moy_mob_total', 'moy_mob_FRAIS', 'moy_mob_FFL', 'moy_mob_GEL', 'month_day']]
    df2021 = df[df['ANNEE']== 1][['moy_mob_total', 'moy_mob_FRAIS', 'moy_mob_FFL', 'moy_mob_GEL','month_day']]

    df_merge=df2021.merge(df2020, on='month_day', how="left")
    df_merge = df_merge.fillna(0)

    df_merge = df_merge[ df_merge["moy_mob_total_y"]!=0]
    df_merge
    plt.figure(figsize=(45,25), dpi = 80)
    #fig = make_subplots(rows=1, cols=1)
    fig1 = go.Figure()
    

    total_2020=go.Scatter (x= df_merge['month_day'], y = df_merge['moy_mob_total_y'],name='total 2020')
    gel_2020=go.Scatter (x= df_merge['month_day'], y = df_merge['moy_mob_FRAIS_y'], name='Frais 2020')
    ffl_2020=go.Scatter (x= df_merge['month_day'], y = df_merge['moy_mob_FFL_y'],name='ffl 2020')
    frais_2020=go.Scatter (x= df_merge['month_day'], y = df_merge['moy_mob_GEL_y'], name='gel 2020')

    total_2021 = go.Scatter (x= df_merge['month_day'], y = df_merge['moy_mob_total_x'],name='total 2021')
    gel_2021=go.Scatter (x= df_merge['month_day'], y = df_merge['moy_mob_FRAIS_x'], name='Frais 2021')
    ffl_2021=go.Scatter (x= df_merge['month_day'], y = df_merge['moy_mob_FFL_x'],name='ffl 2021')
    frais_2021=go.Scatter (x= df_merge['month_day'], y = df_merge['moy_mob_GEL_x'], name='Gel 2021')

    data= [total_2020,gel_2020,ffl_2020,frais_2020,total_2021,gel_2021,ffl_2021,frais_2021]
    fig1.add_traces(data)
    
    
    st.title('Moyenne Mobile Volume par Secteur')
    with st.expander('more information'):
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(mobile, unsafe_allow_html=True)
        with col2:
            st.image('img/mobile2.gif')
    fig1.update_yaxes( title='VOLUME')
    fig1.update_xaxes( title='DATE')
    fig1.update_layout(title='Moyenne Mobile jour 2020/2021',template="simple_white")
    fig1.add_vrect(x0="15", x1="54", 
              annotation_text=" Période forte", annotation_position="top left",
              fillcolor="green", opacity=0.20, line_width=0)
    fig1.add_vrect(x0="55", x1="75", 
              annotation_text=" Période creuse", annotation_position="top left",
              fillcolor="red", opacity=0.20, line_width=0)
    fig1.add_vrect(x0="115", x1="130", 
              annotation_text=" Période creuse", annotation_position="top left",
              fillcolor="red", opacity=0.20, line_width=0)
    fig1.update_xaxes(ticklabelmode="period")
    fig1.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [1,  15,30,50,70,90,110,130,150,170],
        ticktext = ['MARS',  'AVRIL', 'MAI','JUIN','JUILLET','AOUT','SEPTEMBRE','OCTOBRE','DECEMBRE']
    )
)
    fig1.update_layout(hovermode="x unified")
    st.write(fig1)
    
    
    
    st.title('Distribution des volumes par secteur')
    with st.expander('more information'):
        st.write(distrib, unsafe_allow_html=True)
        
    
    x1 = df['REALISE_TOTAL_FRAIS']
    x2 = df['REALISE_TOTAL_GEL']
    x3 = df['REALISE_TOTAL_FFL']
    x4 = df['REALISE_TOTAL_FRAIS']+df['REALISE_TOTAL_GEL']+df['REALISE_TOTAL_FFL']
    hist_data= [x1,x2,x3]
    group_labels = ['Frais','GEL','FFL']

    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=x1,name='FRAIS',nbinsx=30))
    fig4.add_trace(go.Histogram(x=x2,name='GEL',nbinsx=20))
    fig4.add_trace(go.Histogram(x=x3,name='FFL',nbinsx=20))
    fig4.add_trace(go.Histogram(x=x4,name='total site',nbinsx=30))
    fig4.update_traces (opacity=0.7)
    fig4.update_layout(barmode='overlay',bargap=0.1,title='Distribution des volumes par secteur')
    fig4.update_layout(width=1400,height=600,hovermode='y')
    fig4.update_xaxes( title='VOLUME')
    fig4.update_yaxes( title='FREQUENCE')
    st.write(fig4)
    
    
    
    st.title('Volume moyen Jour par secteur')
    with st.expander('more information'):
        st.write(violo,  unsafe_allow_html=True)
    menu = st.radio(
    "",
    ("secteur FRAIS", "secteur GEL", "secteur FFL"),
)
    if menu =='secteur FRAIS':
        dataset = dataset.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FFL'], axis = 1)
        y='REALISE_TOTAL_FRAIS'
        REALISE_TARGETED = 'REALISE_TOTAL_FRAIS'
        fig = px.violin(dataset,x= x, y=y, color= x, box = True)
        fig.update_layout(showlegend=True)
        
        fig.update_layout(width=1400,height=600)
        fig.update_yaxes( title='VOLUME')
        st.write(fig)
        
    
    if menu =='secteur GEL':
        dataset = dataset.drop(['REALISE_TOTAL_FRAIS','REALISE_TOTAL_FFL'], axis = 1)
        y='REALISE_TOTAL_GEL'
        REALISE_TARGETED = 'REALISE_TOTAL_GEL'
        fig = px.violin(dataset,x= x, y=y, color= x, box = True)
        
        fig.update_layout(width=1400,height=600)
        fig.update_yaxes( title='VOLUME')
        st.write(fig)
        
    if menu =='secteur FFL':
        dataset = dataset.drop(['REALISE_TOTAL_GEL','REALISE_TOTAL_FRAIS'], axis = 1)
        y='REALISE_TOTAL_FFL'
        REALISE_TARGETED = 'REALISE_TOTAL_FFL'
        fig = px.violin(dataset,x= x, y=y, color= x, box = True)
        fig.update_yaxes( title='VOLUME')
        fig.update_layout(width=1400,height=600)
        st.write(fig)
        
    
    st.title('Impact Jour Férié')
    with st.expander('more information'):
        st.write(impact, unsafe_allow_html=True)    
        
    jour_ferié()
    matricecorr()





def violon(dataset,x,y):
    fig = px.violin(dataset,x= x, y=x, color= x, box = True)
    return fig
    


def jour_ferié():
    df = load_csv(path)
    cols_to_keep=[
        'ANNEE',
        'MOIS',
        'weekday',
        'REALISE_TOTAL_FFL',
        'REALISE_TOTAL_FRAIS',
        'REALISE_TOTAL_GEL',
        'SEMAINE',
        'dernier_jour_ferie',
        'dernier_jour_ferie_nom',
        'prochain_jour_ferie',
        'prochain_jour_ferie_nom',
        'prox_jour_ferie',
        'prox_jour_ferie_nom'
        ]
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    menu = st.radio(
    "",
    ("secteur frais", "secteur Gel", "secteur FFL"),
)
    if menu =='secteur frais':
        
        REALISE_TARGETED = 'REALISE_TOTAL_FRAIS'
        
        
    
    if menu == 'secteur Gel':
        
        REALISE_TARGETED = 'REALISE_TOTAL_GEL'
        
        
    if menu == 'secteur FFL':
        
        REALISE_TARGETED = 'REALISE_TOTAL_FFL'
        
    

    df_short= df[cols_to_keep]
    
    
    mean_real = df_short[(np.abs(df_short['prox_jour_ferie']) > 8)]

    mean_real = mean_real.groupby('weekday').agg({ REALISE_TARGETED : 'mean'})

    mean_real = mean_real.reset_index()
    meilleur_ecart_montant = [0 for i in range(0, 6)]
    meilleur_ecart_jour = [0 for i in range(0, 6)]
    percent_ecart_list = [0 for i in range(0, 6)]
    list_ecart_per_day = [[0] * 9 for i in range(6)]
    list_ecart_per_day_past = [[0] * 9 for i in range(6)]
    
    for i in range(1, 10) :
# Jour férié prochain
        df_closed_holyday = df_short[df_short['prochain_jour_ferie']<=i]
        mean_real_df_closed_holyday = df_closed_holyday.groupby('weekday').agg({ REALISE_TARGETED : 'mean'})
        mean_real_df_closed_holyday.reset_index()
        ecart_df = mean_real_df_closed_holyday - mean_real
    
        for j in range(0, 6):
            #         print(i)
            ecart = ecart_df.iloc[j][REALISE_TARGETED]
            list_ecart_per_day[j][i-1] = ecart
            #         print(ecart_df)
            #         print(meilleur_ecart[j])
            if (ecart >  meilleur_ecart_montant[j]) :
                meilleur_ecart_montant[j] = ecart
                meilleur_ecart_jour[j] = i

# Jour férié passé   
        df_closed_holyday_past = df_short[(np.abs(df_short['dernier_jour_ferie'])<=i)]
        mean_real_df_closed_holyday_past = df_closed_holyday_past.groupby('weekday').agg({ REALISE_TARGETED : 'mean'})
        mean_real_df_closed_holyday_past.reset_index()
        ecart_df_past = mean_real_df_closed_holyday_past - mean_real
    
        for j in range(0, 6):
            ecart_past = ecart_df_past.iloc[j][REALISE_TARGETED]
            list_ecart_per_day_past[j][i-1] = ecart_past


    x = [i for i in range(1, 10)]
    i = 1

    print(REALISE_TARGETED)
    print("variation du CA à la proximité d'un prochain jour férié quand le jour de la semaine est :")

    fig2 = make_subplots(rows=2,cols=3,subplot_titles=('Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi'),x_title='nbre de jour avant le férié',y_title='delta vs moyenne du jour')
    fig2.update_layout(showlegend=False)
    fig2.update_layout(width=1400,height=700)
    for day_num_ecarts in list_ecart_per_day:
        
        y = day_num_ecarts
        
        
    
    
        if i==1:  
            plt.title("Lundi")
            fig2.add_trace(go.Bar(x=x,y=y ,name='lundi'),row=1 ,col=1 )
        if i==2:  
            plt.title("Mardi")
            fig2.add_trace(go.Bar(x=x,y=y ,name='mardi'),row=1 ,col=2)
        if i==3:  
            plt.title("Mercredi")
            fig2.add_trace(go.Bar(x=x,y=y ,name='mercredi'),row=1 ,col=3)
        if i==4:  
            plt.title("Jeudi")
            fig2.add_trace(go.Bar(x=x,y=y ,name='jeudi' ),row=2 ,col=1)
        if i==5:  
            plt.title("Vendredi")
            fig2.add_trace(go.Bar(x=x,y=y ,name='vendredi'),row=2 ,col=2)
        if i==6:  
            plt.title("Samedi")
            fig2.add_trace(go.Bar(x=x,y=y ,name='samedi' ),row=2 ,col=3)
        i += 1

    
    x = [i for i in range(1, 10)]
    i = 1
    if menu =='secteur frais':
        fig2.update_yaxes(range = (-500,25000))
    if menu =='secteur Gel':
        fig2.update_yaxes(range = (-1000,4200))
    if menu =='secteur FFL':
        fig2.update_yaxes(range = (-6000,10000))
    st.write(fig2)



def matricecorr():
    datacorr = load_csv(path)
    st.title('Matrice de corrélation par secteur')
    
       # st.write(violo,  unsafe_allow_html=True)
    menu = st.radio(
    "",
    ("secteur MECA", "secteur -18°c", "secteur fruits et légumes"))
    
    
    
    if menu =='secteur MECA':
        suppr=['PREVISION_BUDGET_FRAIS',
               'NB_HEURES_TOTAL_FRAIS', 'OBJECTIF_PROD_FRAIS',
       'PREVISION_BUDGET_GEL', 'REALISE_TOTAL_GEL', 'NB_HEURES_TOTAL_GEL',
       'OBJECTIF_PROD_GEL', 'PREVISION_BUDGET_FFL', 'REALISE_TOTAL_FFL',
       'NB_HEURE_PREPARATION_FFL', 'OBJECTIF_PROD_FFL', 'REALISE_GEL_EXP',
       'REALISE_FLF_EXP', 'REALISE_HGE_EXP', 'REALISE_MECA_EXP',
       'TOTAL_EXPE_EXP', 'NB_HEURE_EXP', 'OBJECTIF_PROD_EXP', 'colIndex',]
        datacorr = datacorr.drop(suppr, axis =1)
        corr = datacorr.corr()
        fig5 = plt.figure(figsize=(50,50))
        sns.heatmap(corr, annot=True, cmap ='viridis')
        plt.style.use("dark_background")
        st.write(fig5)
        
    
    if menu =='secteur -18°c':
        suppr=['PREVISION_BUDGET_FRAIS',
               'NB_HEURES_TOTAL_FRAIS', 'OBJECTIF_PROD_FRAIS',
       'PREVISION_BUDGET_GEL', 'REALISE_TOTAL_FRAIS', 'NB_HEURES_TOTAL_GEL',
       'OBJECTIF_PROD_GEL', 'PREVISION_BUDGET_FFL', 'REALISE_TOTAL_FFL',
       'NB_HEURE_PREPARATION_FFL', 'OBJECTIF_PROD_FFL', 'REALISE_GEL_EXP',
       'REALISE_FLF_EXP', 'REALISE_HGE_EXP', 'REALISE_MECA_EXP',
       'TOTAL_EXPE_EXP', 'NB_HEURE_EXP', 'OBJECTIF_PROD_EXP', 'colIndex',]
        datacorr = datacorr.drop(suppr, axis =1)
        
        corr = datacorr.corr()
        fig5 = plt.figure(figsize=(50,50))
        sns.heatmap(corr, annot=True, cmap ='viridis')
        plt.style.use("dark_background")
        
        st.write(fig5)
        
    if menu =='secteur fruits et légumes':
        suppr=['PREVISION_BUDGET_FRAIS',
               'NB_HEURES_TOTAL_FRAIS', 'OBJECTIF_PROD_FRAIS',
       'PREVISION_BUDGET_GEL', 'REALISE_TOTAL_GEL', 'NB_HEURES_TOTAL_GEL',
       'OBJECTIF_PROD_GEL', 'PREVISION_BUDGET_FFL', 'REALISE_TOTAL_FRAIS',
       'NB_HEURE_PREPARATION_FFL', 'OBJECTIF_PROD_FFL', 'REALISE_GEL_EXP',
       'REALISE_FLF_EXP', 'REALISE_HGE_EXP', 'REALISE_MECA_EXP',
       'TOTAL_EXPE_EXP', 'NB_HEURE_EXP', 'OBJECTIF_PROD_EXP', 'colIndex',]
        datacorr = datacorr.drop(suppr, axis =1)
        
        corr = datacorr.corr()
        fig5 = plt.figure(figsize=(50,50))
        sns.heatmap(corr, annot=True, cmap ='viridis')
        plt.style.use("dark_background")
        st.write(fig5)
    
