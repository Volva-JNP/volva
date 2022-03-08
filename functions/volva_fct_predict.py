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
from datetime import date
import datetime
from functions.utils.functions import proximite_jour_ferie
from functions.utils.jours_feries import get_nom_jour_ferie
from sklearn.preprocessing import StandardScaler 
from stqdm import stqdm

from joblib import load


 
def numOfDays(date1, date2):
    return (date2-date1).days
     
date1 = date(2018, 12, 13)
date2 = date(2019, 2, 25)
print(numOfDays(date1, date2), "days")

from sklearn.model_selection import train_test_split

def build_page_predict():

    st.title('Prédictions sur une période')

    with st.expander('Information'):    
        col1, col2 = st.columns(2)
        with col1:
            st.write("A compléter")
        with col2:
            st.write("A compléter")

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
    


    if menu_secteur!="vide":
        
        if menu_secteur =='secteur frais': 
            secteur="REALISE_TOTAL_FRAIS" 

        elif  menu_secteur =='secteur GEL':
            secteur="REALISE_TOTAL_GEL"   

        elif  menu_secteur =='secteur FFL':
            secteur="REALISE_TOTAL_FFL" 

        button_predict = st.button('Obtenir les prédictions')

        if button_predict:

            df = add_time_datas(date_debut,date_fin)
            df = add_holydays(df)
            df = add_promotions(df)
            df = add_temperatures_data(df)            
            df = selection_data(df,secteur)
            with st.expander('Voir le dataset'):   
                st.write(df)

            df_best_model_per_day = pd.read_csv('datas/' + 'df_best_model_per_day-' + secteur + '.csv')

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

            for model_name in df_best_model_per_day['Modèle'].unique():
                               
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


                # elif model_name == "BR":
                #     model_BR = load('models/model_' + model_name +'_' + secteur + '.joblib')
                #     df_best_model_per_day_BR = df_best_model_per_day[df_best_model_per_day['Modèle']=='BR']
                # elif model_name == "LASSO":
                #     model_LASSO = load('models/model_' + model_name +'_' + secteur + '.joblib')
                #     df_best_model_per_day_LASSO = df_best_model_per_day[df_best_model_per_day['Modèle']=='LASSO']

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

                # # elif model_name == "KNR":
                # #     model_KNR = load('models/model_' + model_name +'_' + secteur + '.joblib')
                # #     df_best_model_per_day_KNR = df_best_model_per_day[df_best_model_per_day['Modèle']=='KNR']
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
            st.write(df_predictions)


            
            
           

def selection_data(df,secteur):

    file_name="df_secteur_best_datas.csv"
    file = 'datas/' + file_name
    df_secteur_best_datas = pd.read_csv(file, header=0)
    df_secteur_best_datas = df_secteur_best_datas[df_secteur_best_datas['secteur']==secteur]
    basis_data = df_secteur_best_datas.iloc[0,3]
    added_data = df_secteur_best_datas.iloc[0,4]
    df = drop_datas(df,basis_data)

    if "F" not in added_data:
        cols_to_drop = [
            'prochain_jour_ferie' ,
            'dernier_jour_ferie'
        ]
        df.drop(cols_to_drop, inplace=True, axis=1)

    if "P" not in added_data:
        df.drop(st.session_state.list_cols_promotions, inplace=True, axis=1)

    if "T" not in added_data:
        cols_to_drop = [
            'Temp moy Auvergne' ,
            'Temp moy Bourgogne'
        ]        
        df.drop(cols_to_drop, inplace=True, axis=1)
    

    cols_to_drop = [
        # 'DATE',
        'JOUR',
        'MOIS',
        # 'weekday',
        'SEMAINE' ,
        'index'
    ]

    df.drop(cols_to_drop, inplace=True, axis=1)
    return df


def drop_datas(df,drop_list):
    if 'A_' not in  drop_list:
        df=df.drop(['ANNEE'], axis=1)
    if 'MP_' not in  drop_list:
        df=df.drop(['month_period'], axis=1)

    if 'MD_' not in  drop_list:
        df=df.drop(['monthday_sin', 'monthday_cos'], axis=1)

    if 'WD_' not in  drop_list:
        df=df.drop(['weekday_sin','weekday_cos'], axis=1)

    if 'M_' not in  drop_list:
        df=df.drop(['mois_sin','mois_cos'], axis=1)

    if 'W_' not in  drop_list:
        df=df.drop(['semaine_cos','semaine_sin'], axis=1)
        
    return df



def add_time_datas(date_debut,date_fin):
    nb_days = numOfDays(date_debut, date_fin)
    dates_range=[date_debut]
    temp_date =date_debut
    for i in range(nb_days):
        new_date = temp_date + datetime.timedelta(days=1)
        temp_date=new_date
        dates_range.append(new_date)
        i+=1

    df = pd.DataFrame(dates_range, columns=['DATE'])
    df['JOUR']=df['DATE'].apply(lambda date : date.day)
    df['weekday'] = df['DATE'].apply(lambda date : get_weekday(date))
    df['MOIS'] = df['DATE'].apply(lambda date : date.month)
    df['month_period'] = df['DATE'].apply(lambda row : 1 if row.day<=5 else 2 if row.day>=26 else 0 )
    df['SEMAINE'] = df['DATE'].apply(lambda date : date.isocalendar()[1])
    df['ANNEE'] = df['DATE'].apply(lambda date : date.year)
    
    df['ANNEE']=df['ANNEE'].replace({2020:0,2021:1,2022:2})


    df['weekday_sin'] = np.sin(df['weekday'] * 2 * np.pi / 7)
    df['weekday_cos'] = np.cos(df['weekday'] * 2 * np.pi / 7)

    df['monthday_sin'] = np.sin(df['JOUR'] * 2 * np.pi / 31)
    df['monthday_cos'] = np.cos(df['JOUR'] * 2 * np.pi / 31)

    df['mois_sin'] = np.sin(df['MOIS'] * 2 * np.pi / 12)
    df['mois_cos'] = np.cos(df['MOIS'] * 2 * np.pi / 12)

    df['semaine_sin'] = np.sin(df['SEMAINE'] * 2 * np.pi / 52)
    df['semaine_cos'] = np.cos(df['SEMAINE'] * 2 * np.pi / 52)



    return df
    

def add_holydays(df):
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)  
    df['dernier_jour_ferie'] = df.apply(lambda row: -proximite_jour_ferie(row['DATE'], 'last')[0]  , axis = 1)
    df['prochain_jour_ferie'] = df.apply(lambda row: proximite_jour_ferie(row['DATE'], 'next')[0]  , axis = 1)
    
    return df


def add_promotions(df):

    file_name="promo_clean.xls"
    file = 'datas/' + file_name
    promotions = pd.read_excel(file, header=0)
    promotions['DLC'] = promotions['DLC'].fillna('no_dlc') 
    promotions['vitesse'] = promotions['DLC'].fillna('no_speed') 
    promotions['Code_1'] = promotions['Code_1'].fillna('no_code') 
    promotions.info()
    list_cols_promotions = [
        "NETTO",
        "GEL",
        "no_dlc",
        "no_code",
        "AFG",
        " 46-001",
        "comp",
        "OPE",
        "INTER",
        " 47-001",
        "PUB",
        "SEC",
        "AFS",
        " 41/52-001",
        "FRAIS",
        "TD",
        "AFF",
        " 43/45/67-001",
        "MEA",
        " 48-001",
        "DLC Longue",
        "TA",
        " 43-002",
        "DLC Courte",
        " 49-004",
        " 49-001",
        " 43-007",
        " 49-007",
        " 49-002",
        "REA",
        " 40-001/41-018",
        " 10-016/40-012/41-006",
        " 41-011/43-019",
        ]
    for col in list_cols_promotions:
        df[col]=0

    for i in df.index: 
        date_jour = df["DATE"][i]
        for j in promotions.index: 
    #         print(i,'-',j)
            date_debut_ope = promotions["Date Début Comm"][j]
            date_fin_ope = promotions["Date Fin Opé"][j]
            
            if   date_debut_ope <= date_jour <= date_fin_ope : 
                enseigne = promotions["Enseigne"][j]            
                secteur = promotions["Secteur"][j]   
                DLC = promotions["DLC"][j]
                vitesse = promotions["vitesse"][j]
                Code_1 = promotions["Code_1"][j]   
                Code_2 = promotions["Code_2"][j]
                Code_3 = promotions["Code_3"][j] 
                Pub = promotions["Pub"][j]   
                Type = promotions["Type"][j]
                
                
                # if enseigne not in df.columns: 
                #     df[enseigne]=0  
                #     list_cols_promotions.append(enseigne)              
                # if secteur not in df.columns: 
                #     df[secteur]=0
                #     list_cols_promotions.append(secteur)  
                # if DLC not in df.columns: 
                #     df[DLC]=0
                #     list_cols_promotions.append(DLC)  
                # if vitesse not in df.columns: 
                #     df[vitesse]=0
                #     list_cols_promotions.append(vitesse)  
                # if Code_1 not in df.columns: 
                #     df[Code_1]=0
                #     list_cols_promotions.append(Code_1)  
                # if Code_2 not in df.columns: 
                #     df[Code_2]=0
                #     list_cols_promotions.append(Code_2)  
                # if Code_3 not in df.columns: 
                #     df[Code_3]=0
                #     list_cols_promotions.append(Code_3)  
                # if Pub not in df.columns: 
                #     df[Pub]=0
                #     list_cols_promotions.append(Pub)  
                # if Type not in df.columns: 
                #     df[Type]=0
                #     list_cols_promotions.append(Type)  
                    
                    
                df[enseigne][i] = df[enseigne][i] + 1
                df[secteur][i] = df[enseigne][i] + 1
                df[DLC][i] = df[DLC][i] + 1  
                df[vitesse][i] = df[vitesse][i] + 1  
                df[Code_1][i] = df[Code_1][i] + 1  
                df[Code_2][i] = df[Code_2][i] + 1  
                df[Code_3][i] = df[Code_3][i] + 1  
                df[Pub][i] = df[Pub][i] + 1  
                df[Type][i] = df[Type][i] + 1 

        st.session_state.list_cols_promotions = list_cols_promotions     


    return df



def add_temperatures_data(df):
    
    temperatures=pd.read_csv('datas/temperature-quotidienne-regionale.csv', sep = ';')
    temperatures[["ANNEE", "MOIS", "JOUR"]] = temperatures["Date"].str.split("-", expand = True)
    temperatures["MOIS"] = temperatures["MOIS"].astype(int)
    temperatures["JOUR"] = temperatures["JOUR"].astype(int)
    temperatures_ARA = temperatures[(temperatures['Région'] == 'Auvergne-Rhône-Alpes')]
    temperatures_BFC = temperatures[(temperatures['Région'] == 'Bourgogne-Franche-Comté')]

    state_summary_ARA = temperatures_ARA.groupby(['MOIS','JOUR']).agg({"TMoy (°C)": 'mean'})
    state_summary_ARA = state_summary_ARA.reset_index()
    state_summary_BFC = temperatures_BFC.groupby(['MOIS','JOUR']).agg({"TMoy (°C)": 'mean'})
    state_summary_BFC = state_summary_BFC.reset_index()
    df_temp_moy = state_summary_ARA.merge(state_summary_BFC, on =['MOIS', 'JOUR'],  how ='inner')
    df_temp_moy = df_temp_moy.reset_index()
    df = df.merge(df_temp_moy, on =['MOIS', 'JOUR'],  how ='left')

    dictionnaire = {'TMoy (°C)_x': 'Temp moy Auvergne',
                'TMoy (°C)_y': 'Temp moy Bourgogne'}
    df = df.rename(dictionnaire, axis = 1)

    return df
    


   


def get_weekday(date) :
    return date.weekday()
