from calendar import weekday
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools
from texts.volva_text import *
from plotly.subplots import make_subplots
from functions.volva_fct import *
from functions.utils.functions import *
from functions.utils.jours_feries import get_nom_jour_ferie
from sklearn.preprocessing import StandardScaler 
from stqdm import stqdm

from joblib import load





def build_page_predict():

    st.title('Prévisions sur une période')

    with st.expander('Information'):    
        col1, col2 = st.columns(2)
        with col1:
            st.write("")
            st.write("")
            st.write(periode)
        with col2:
            st.image("img/delivery.gif")

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    menu_secteur  = st.radio(    
    "",
    ("vide", "secteur frais", "secteur GEL", "secteur FFL"),

    )        

    col_from, col_to = st.columns(2)
    with col_from:
        date_debut = st.date_input('Date de début')
    with col_to:
        date_fin = st.date_input('Date de fin')
    


    if menu_secteur!="vide" :
       
        if menu_secteur == "secteur frais":
            secteur="REALISE_TOTAL_FRAIS"

        elif  menu_secteur =='secteur GEL':
            secteur="REALISE_TOTAL_GEL"   

        elif  menu_secteur =='secteur FFL':
            secteur="REALISE_TOTAL_FFL" 

        df_secteur_best_datas = pd.read_csv('datas/' + 'df_secteur_best_datas.csv')  
        df_secteur_best_datas_secteur =  df_secteur_best_datas[df_secteur_best_datas['secteur']==secteur]  
        score = df_secteur_best_datas_secteur.iloc[0,1]

        if score < 0.70 :
            st.warning("Les modèles ne sont pas assez performants sur le secteur frais pour permettre des prévisions ...")

        else:

            button_predict = st.button('Obtenir les prévisions')

            if button_predict:
                placeholder = st.empty()
                placeholder.warning("Le dataset est en cours de préparation. Veuillez patienter ...")
                df = pd.DataFrame()
                df = add_time_datas(date_debut,date_fin,df)
                df = add_holydays(df)
                df = add_promotions(df)
                df = add_temperatures_data(df)            
                df = selection_data(df,secteur)
                df = drop_time_and_index_fields(df)
                placeholder.empty()
                with st.expander('Voir le dataset'):   
                    st.write(df)

                df_best_model_per_day = pd.read_csv('datas/' + 'df_best_model_per_day-'+ secteur +'.csv')  

                dict_weekday_num = {

                        'Lun' : 0,
                        'Mar' : 1,                    
                        'Mer' : 2,
                        'Jeu' : 3,
                        'Ven' : 4,
                        'Sam' : 5

                }
                df_best_model_per_day= df_best_model_per_day.replace(dict_weekday_num)

    
                df_predictions = pd.DataFrame();
                list_models = df_best_model_per_day['Modèle'].unique()
                nb_model = len(list_models) - 1

                for model_name,i in zip(list_models, stqdm(range(nb_model))):
                                
                    if model_name == "GBR" : 
                        model_GBR = load('models/model_' + model_name +'_' + secteur + '.joblib')
                        df_best_model_per_day_GBR = df_best_model_per_day[df_best_model_per_day['Modèle']=='GBR']

                        for jour in df_best_model_per_day_GBR['Jours'].unique():
                            df_jour = df[df['weekday'] == jour]
                            
                            if df_jour.shape[0] != 0 :
                                df_jour_DATE_WeekDay = df_jour[['DATE', 'weekday']]
                                df_jour = df_jour.drop(['DATE', 'weekday'], axis=1)
                                scaler = StandardScaler().fit(df_jour)
                                df_jour_scaled = scaler.transform(df_jour)
                                predictions_GBR = model_GBR.predict(df_jour_scaled)
                                df_predictions_GBR = df_jour_DATE_WeekDay['DATE'].reset_index()
                                df_predictions_GBR = df_predictions_GBR.drop('index', axis=1)
                                df_predictions_GBR  = pd.concat([df_predictions_GBR,pd.Series(predictions_GBR)], axis=1)
                                df_predictions = pd.concat([df_predictions,df_predictions_GBR], axis=0)                           


                    elif model_name == "BR":
                        model_BR = load('models/model_' + model_name +'_' + secteur + '.joblib')
                        df_best_model_per_day_BR = df_best_model_per_day[df_best_model_per_day['Modèle']=='BR']
                        for jour in df_best_model_per_day_BR['Jours'].unique():
                            df_jour = df[df['weekday'] == jour]                        
                            if df_jour.shape[0] != 0 :
                                df_jour_DATE_WeekDay = df_jour[['DATE', 'weekday']]
                                df_jour = df_jour.drop(['DATE', 'weekday'], axis=1)
                                scaler = StandardScaler().fit(df_jour)
                                df_jour_scaled = scaler.transform(df_jour)
                                predictions_BR = model_BR.predict(df_jour_scaled)
                                df_predictions_BR = df_jour_DATE_WeekDay['DATE'].reset_index()
                                df_predictions_BR = df_predictions_BR.drop('index', axis=1)
                                df_predictions_BR  = pd.concat([df_predictions_BR,pd.Series(predictions_BR)], axis=1)
                                df_predictions = pd.concat([df_predictions,df_predictions_BR], axis=0) 

                    elif model_name == "LASSO":
                        model_LASSO = load('models/model_' + model_name +'_' + secteur + '.joblib')
                        df_best_model_per_day_LASSO = df_best_model_per_day[df_best_model_per_day['Modèle']=='LASSO']
                        for jour in df_best_model_per_day_LASSO['Jours'].unique():
                            if df_jour.shape[0] != 0 :
                                df_jour_DATE_WeekDay = df_jour[['DATE', 'weekday']]
                                df_jour = df_jour.drop(['DATE', 'weekday'], axis=1)
                                scaler = StandardScaler().fit(df_jour)
                                df_jour_scaled = scaler.transform(df_jour)
                                predictions_LASSO = model_LASSO.predict(df_jour_scaled)
                                df_predictions_LASSO = df_jour_DATE_WeekDay['DATE'].reset_index()
                                df_predictions_LASSO = df_predictions_LASSO.drop('index', axis=1)
                                df_predictions_LASSO  = pd.concat([df_predictions_LASSO,pd.Series(predictions_LASSO)], axis=1)
                                df_predictions = pd.concat([df_predictions,df_predictions_LASSO], axis=0) 

                    elif model_name == "RFR":
                        model_RFR = load('models/model_' + model_name +'_' + secteur + '.joblib')
                        df_best_model_per_day_RFR = df_best_model_per_day[df_best_model_per_day['Modèle']=='RFR']
                        for jour in df_best_model_per_day_RFR['Jours'].unique():
                            df_jour = df[df['weekday'] == jour]
                            if df_jour.shape[0] != 0 :
                                df_jour_DATE_WeekDay = df_jour[['DATE', 'weekday']]
                                df_jour = df_jour.drop(['DATE', 'weekday'], axis=1)
                                scaler = StandardScaler().fit(df_jour)
                                df_jour_scaled = scaler.transform(df_jour)
                                predictions_RFR = model_RFR.predict(df_jour_scaled)
                                df_predictions_RFR = df_jour_DATE_WeekDay['DATE'].reset_index()
                                df_predictions_RFR = df_predictions_RFR.drop('index', axis=1)
                                df_predictions_RFR  = pd.concat([df_predictions_RFR,pd.Series(predictions_RFR)], axis=1)
                                df_predictions = pd.concat([df_predictions,df_predictions_RFR], axis=0)

                    elif model_name == "KNR":
                        model_KNR = load('models/model_' + model_name +'_' + secteur + '.joblib')
                        df_best_model_per_day_KNR = df_best_model_per_day[df_best_model_per_day['Modèle']=='KNR']
                        for jour in df_best_model_per_day_KNR['Jours'].unique():
                                df_jour = df[df['weekday'] == jour]
                                if df_jour.shape[0] != 0 :
                                    df_jour_DATE_WeekDay = df_jour[['DATE', 'weekday']]
                                    df_jour = df_jour.drop(['DATE', 'weekday'], axis=1)
                                    scaler = StandardScaler().fit(df_jour)
                                    df_jour_scaled = scaler.transform(df_jour)
                                    predictions_KNR = model_KNR.predict(df_jour_scaled)
                                    df_predictions_KNR = df_jour_DATE_WeekDay['DATE'].reset_index()
                                    df_predictions_KNR = df_predictions_KNR.drop('index', axis=1)
                                    df_predictions_KNR  = pd.concat([df_predictions_KNR,pd.Series(predictions_KNR)], axis=1)
                                    df_predictions = pd.concat([df_predictions,df_predictions_KNR], axis=0) 

                    elif model_name == "EN":
                        model_EN = load('models/model_' + model_name +'_' + secteur + '.joblib')
                        df_best_model_per_day_EN = df_best_model_per_day[df_best_model_per_day['Modèle']=='EN']
                        for jour in df_best_model_per_day_EN['Jours'].unique():
                                    df_jour = df[df['weekday'] == jour]
                                    if df_jour.shape[0] != 0 :
                                        df_jour_DATE_WeekDay = df_jour[['DATE', 'weekday']]
                                        df_jour = df_jour.drop(['DATE', 'weekday'], axis=1)
                                        scaler = StandardScaler().fit(df_jour)
                                        df_jour_scaled = scaler.transform(df_jour)
                                        predictions_EN = model_EN.predict(df_jour_scaled)
                                        df_predictions_EN = df_jour_DATE_WeekDay['DATE'].reset_index()
                                        df_predictions_EN = df_predictions_DTR.drop('index', axis=1)
                                        df_predictions_EN  = pd.concat([df_predictions_DTR,pd.Series(predictions_EN)], axis=1)
                                        df_predictions = pd.concat([df_predictions,df_predictions_EN], axis=0)

                    elif model_name == "DTR":                
                        model_DTR = load('models/model_' + model_name +'_' + secteur + '.joblib')
                        df_best_model_per_day_DTR = df_best_model_per_day[df_best_model_per_day['Modèle']=='DTR']
                        for jour in df_best_model_per_day_DTR['Jours'].unique():
                                df_jour = df[df['weekday'] == jour]
                                if df_jour.shape[0] != 0 :
                                    df_jour_DATE_WeekDay = df_jour[['DATE', 'weekday']]
                                    df_jour = df_jour.drop(['DATE', 'weekday'], axis=1)
                                    scaler = StandardScaler().fit(df_jour)
                                    df_jour_scaled = scaler.transform(df_jour)
                                    predictions_DTR = model_DTR.predict(df_jour_scaled)
                                    df_predictions_DTR = df_jour_DATE_WeekDay['DATE'].reset_index()
                                    df_predictions_DTR = df_predictions_DTR.drop('index', axis=1)
                                    df_predictions_DTR  = pd.concat([df_predictions_DTR,pd.Series(predictions_DTR)], axis=1)
                                    df_predictions = pd.concat([df_predictions,df_predictions_DTR], axis=0)
                
                df_predictions = df_predictions.sort_values(by=['DATE'])
                df_predictions['Prévisions ETP'] = (np.ceil(df_predictions[0]/1000)).astype(int)
                df_predictions = df_predictions.reset_index()
                df_predictions.drop([0, 'index'], axis=1, inplace=True)
                df_predictions['DATE'] = df_predictions['DATE'].dt.strftime('%d-%m-%Y')
                col1, col2 = st.columns(2)
                with col1:
                    st.write("")
                    st.write("Tableau de prévisions")
                    st.write("")
                    st.write("")
                    st.write(df_predictions)
                with col2:
                    
                    # fig9=px.line(data_frame=df_predictions,x='DATE',y='Prévisions ETP',text='Prévisions ETP', markers=False)           
                    fig9=px.line(data_frame=df_predictions,x='DATE',y='Prévisions ETP',text='Prévisions ETP')
                    fig9.update_yaxes(dtick=1)
                    fig9.update_traces(textposition='top center')
                    fig9.update_layout(title_text="Graphique des prévisions (en nombre d'opérateurs)")

                    st.write(fig9)

    
        


            
            
