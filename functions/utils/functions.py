############## construct_month_df() : Fonction qui constuit un dataframe pour le fichier de données
############## passé en paramêtre en selectionnant les colonnes qui nous intéressent
############## Fields liste les colonnes que l'on souhaite conserver

import datetime
import pandas as pd
import numpy as np
from calendar import monthrange
from functions.utils import jours_feries
from datetime import datetime 
from datetime import timedelta
import plotly.express as px
import plotly.graph_objs as go
from texts.volva_text import *
from plotly.subplots import make_subplots
from functions.volva_fct import *
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, train_test_split , cross_val_score, StratifiedKFold, cross_val_predict, cross_validate
from math import *
from sklearn.ensemble import GradientBoostingRegressor
from stqdm import stqdm
from joblib import dump

GBR = GradientBoostingRegressor() # Params pour GEL
params_gbr = {    
        'max_depth': [1], 
#         'n_estimators': [i for i in np.arange(1100, 1500,100 )],
#         'learning_rate': [i for i in np.arange(0.0095, 0.010,0.0001 )]
        'n_estimators': [1400],
        'learning_rate': [0.099]
}

path = 'datas/volva_datas_utlimate_one.csv'

def redim_df():
    df= load_csv(path)
    suppr=[
            'MOIS',
            'SEMAINE',
            'JOUR',
            # 'DATE',
            # 'weekday',
            'monthdays',
            'prox_jour_ferie' ,    
            'PREVISION_BUDGET_FRAIS',
            'NB_HEURES_TOTAL_FRAIS',
            'OBJECTIF_PROD_FRAIS',
            'PREVISION_BUDGET_GEL',
            'NB_HEURES_TOTAL_GEL',
            'OBJECTIF_PROD_GEL',
            'PREVISION_BUDGET_FFL',
            'NB_HEURE_PREPARATION_FFL',
            'OBJECTIF_PROD_FFL',
            'REALISE_FLF_EXP',
            'REALISE_HGE_EXP',
            'REALISE_MECA_EXP',
            'TOTAL_EXPE_EXP',
            'NB_HEURE_EXP',
            'OBJECTIF_PROD_EXP',
            'colIndex',
            'REALISE_GEL_EXP', 
            'dernier_jour_ferie_nom', 
            'prochain_jour_ferie_nom', 
            'prox_jour_ferie_nom',
            'Sem Juin',
            'Sem aout',
            'Sem dec',                  
        ]

    df_total = df 

    
    df = df.drop(suppr, axis = 1)
    df.to_csv('datas/volva_datas_utlimate_one2.csv')
    

    return df


def get_df():
    df= load_csv(path)
    suppr=[
            'MOIS',
            'SEMAINE',
            'JOUR',
            'DATE',
            'weekday',
            'monthdays',
            'prox_jour_ferie' ,    
            'PREVISION_BUDGET_FRAIS',
            'NB_HEURES_TOTAL_FRAIS',
            'OBJECTIF_PROD_FRAIS',
            'PREVISION_BUDGET_GEL',
            'NB_HEURES_TOTAL_GEL',
            'OBJECTIF_PROD_GEL',
            'PREVISION_BUDGET_FFL',
            'NB_HEURE_PREPARATION_FFL',
            'OBJECTIF_PROD_FFL',
            'REALISE_FLF_EXP',
            'REALISE_HGE_EXP',
            'REALISE_MECA_EXP',
            'TOTAL_EXPE_EXP',
            'NB_HEURE_EXP',
            'OBJECTIF_PROD_EXP',
            'colIndex',
            'REALISE_GEL_EXP', 
            'dernier_jour_ferie_nom', 
            'prochain_jour_ferie_nom', 
            'prox_jour_ferie_nom',
            'Sem Juin',
            'Sem aout',
            'Sem dec',                  
        ]

    df_total = df     
    df = df.drop(suppr, axis = 1)    
    df_minimum = pd.concat([df.iloc[:, :5],df.iloc[:, 40:48]], axis=1)

    return df, df_total, df_minimum




def construct_month_df(columns_list, df, month, year):   

    from utils import fields_module

    columns_list = np.asarray(columns_list)
    columns_list[ columns_list == 'ï»¿1'] = '1' # change all occurrences of 8 by 0
    columns_list = columns_list.tolist()
    fields = fields_module.get_fields()
    list_jours_feries = jours_feries.liste_jours_feries_date()
    # print(list_jours_feries)
    # print(columns_list)

    month_df = pd.DataFrame()

    for index, field in fields.items():  

        num_field = index  
        # print(num_field)
        
        if num_field in columns_list:
            column_num = columns_list.index(num_field)
            month_df[field] = df.iloc[:,column_num]
            # print(column_num)

            if month_df[field].dtypes == object and field !='DATE':
                try:
                   serie_month_df_temp = month_df[field].astype(str).str.replace(' ','')
                   serie_month_df_temp = serie_month_df_temp.str.replace('-','0')
                   serie_month_df_temp = serie_month_df_temp.str.replace(',','.')
                   month_df[field] = serie_month_df_temp
                except Exception as e:                        
                    print(e)
        else:
            month_df[field] = ''
            
    month_df['DATE'] = pd.to_datetime(month_df['DATE'], dayfirst=True)    
    month_df = month_df[ (month_df['DATE'].isna() == False)]    
    month_df = month_df[ (month_df['JOUR'] != 'dim') &   ~month_df['DATE'].isin(list_jours_feries)]     
    # print(month_df)
    return month_df



def proximite_jour_ferie(jour, param='next'):
    
    # print(jour)
    # jour = datetime.strptime(jour, "%Y-%m-%d")
    # jour = datetime.fromtimestamp(jour)
    list_jours_feries = jours_feries.liste_jours_feries()
    list_nb_days_from_next_holydays=[]
    for jour_ferie_nom_et_date in list_jours_feries: 
        jours_feries_split = jour_ferie_nom_et_date.split(':')
        jour_ferie=jours_feries_split[0]
        jour_ferie  = datetime.strptime(jour_ferie, "%Y-%m-%d")

        if param == 'next':
            if jour_ferie > jour :
                list_nb_days_from_next_holydays.append((jour_ferie-jour).days)
                

        if param == 'last':
            if jour_ferie < jour :
                list_nb_days_from_next_holydays.append((jour-jour_ferie).days)

    list_nb_days_from_next_holydays.sort()

    # print(jour)
    # print(list_nb_days_from_next_holydays)
    # print('--------------------------------------------------')


    nb_jour = list_nb_days_from_next_holydays[0]

    if (param=='next'):

        jour_ferie_prox = jour + timedelta(days=nb_jour)
    else :
        jour_ferie_prox = jour - timedelta(days=nb_jour)

    jour_ferie_prox  = datetime.strftime(jour_ferie_prox, "%Y-%m-%d")

    info_jour_ferie=[nb_jour,jour_ferie_prox]

    return info_jour_ferie

    
   
def test_model(df_kept,Model,params,secteur, nom_model):

    gridcv_model = 0
    y_test = 0
    pred_test = 0

    if 'y_test' + "_" + secteur in st.session_state and 'pred_test_' + nom_model + "_" + secteur  in st.session_state:
        gridcv_model = st.session_state['model_' + nom_model + "_" + secteur] 
        y_test  = st.session_state['y_test' + "_" + secteur] 
        pred_test  = st.session_state['pred_test_' + nom_model + "_" + secteur] 
        score_test = st.session_state['score_test_' + nom_model + "_" + secteur]
        score_train = st.session_state['score_train_' + nom_model + "_" + secteur]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score train",score_train)
        with col2:
            st.metric("Score test",score_test)   

    else :
        placeholder_button = st.empty()
        button_test_model = placeholder_button.button('Test modèle ' + nom_model)
        if button_test_model:
            placeholder2 = st.empty()
            placeholder2.warning("Veuillez patienter pendant l'évaluation du modèle ...")
            gridcv_model, X_train_scaled, X_test_scaled, y_train, y_test = train_model(df_kept,Model,params,secteur)
            score_train = np.round(gridcv_model.score(X_train_scaled, y_train),4)
            score_test =  np.round(gridcv_model.score(X_test_scaled, y_test),4)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score train",score_train)
            with col2:
                st.metric("Score test",score_test)   
            
            placeholder2.empty()    
            placeholder_button.empty()        

            pred_test = gridcv_model.predict(X_test_scaled) 
            # st.write(pd.DataFrame(pred_test))

            st.session_state['score_test_' + nom_model + "_" + secteur ] = score_test
            st.session_state['score_train_' + nom_model + "_" + secteur] = score_train
            st.session_state['model_' + nom_model + "_" + secteur] = gridcv_model
            st.session_state['y_test' + "_" + secteur] = y_test
            st.session_state['pred_test_' + nom_model + "_" + secteur ] = pred_test 

    return gridcv_model, y_test, pred_test

def display_model_comparaison(list_ecart, list_nom_model, type):
        fig = go.Figure()
        df_ecart = pd.DataFrame(columns=['jour', 'ecart', 'model'])
        if type == "MSE" :
            Title = "Ecarts RMSE moyens par jour"
        elif type == "MAE":
            Title = "Ecarts MAE moyens par jour"

        for ecart_nom, ecart_tab in zip(list_nom_model,list_ecart):
            for jour, ecart_value in zip(['Lun', 'Mar', 'Mer', 'Jeu', 'Ven','Sam'],ecart_tab):
                df_ecart = df_ecart.append(
                                {'jour' : jour,
                                'ecart' : ecart_value,
                                'model' : ecart_nom}
                                , ignore_index=True)

        fig = px.bar(df_ecart, x="jour", y="ecart", color='model',  barmode="group", title=Title)
        st.write(fig)

        return df_ecart

def display_test_pred_graph(y_test, pred_test, df_total):
        fig = go.Figure()
        # Add traces
        fig.add_trace(go.Scatter(x=y_test, y=pred_test,
                            marker_color = df_total['weekday'].astype('int'),
                            mode='markers',
                            text=df_total['weekday'],
                            name='Previsions'))

        fig.add_trace(go.Scatter(x=[y_test.min(),y_test.max()], y=[y_test.min(),y_test.max()],
                            mode='lines',
                            name='prediction parfaite'))

        st.write(fig)
        
        

def build_df(df, secteur,drop_list):

    # st.write(df.columns)
    suppr=[]
    if secteur == 'REALISE_TOTAL_FRAIS':
        suppr.append('REALISE_TOTAL_GEL')
        suppr.append('REALISE_TOTAL_FFL')
    elif secteur == 'REALISE_TOTAL_GEL':
        suppr.append('REALISE_TOTAL_FRAIS')
        suppr.append('REALISE_TOTAL_FFL')
    elif secteur == 'REALISE_TOTAL_FFL':
        suppr.append('REALISE_TOTAL_FRAIS')
        suppr.append('REALISE_TOTAL_GEL')

    # st.write(suppr)
    df_FPTV = df.drop(suppr, axis = 1)
    df_FPTV = drop_datas(df_FPTV,drop_list)

    df_min = pd.concat([df.iloc[:, :5],df.iloc[:, 40:48]], axis=1).drop(suppr, axis = 1)
    df_min = drop_datas(df_min,drop_list)


    df_F = df.iloc[:, 5:7]
    df_P = df.iloc[:, 7:40]
    df_T = df.iloc[:, 48:50]
    df_V = df.iloc[:, 50:]


    return df_FPTV, df_min, df_F, df_P, df_V, df_T

def build_list_test(df_FPTV, df_min, df_F, df_P, df_V, df_T):

    list_df = [ 
        df_min,
        df_FPTV,
        concat_df_test(df_min,[df_F]),
        concat_df_test(df_min,[df_F,df_P]),
        concat_df_test(df_min,[df_F,df_T]),
        concat_df_test(df_min,[df_F,df_V]),
        concat_df_test(df_min,[df_F,df_P,df_T]),
        concat_df_test(df_min,[df_F,df_P,df_V]),
        concat_df_test(df_min,[df_F,df_T,df_V]),
        concat_df_test(df_min,[df_P]),
        concat_df_test(df_min,[df_P,df_T]),
        concat_df_test(df_min,[df_P,df_V]),
        concat_df_test(df_min,[df_P,df_T,df_V]),
        concat_df_test(df_min,[df_T]),
        concat_df_test(df_min,[df_T,df_V]),
        concat_df_test(df_min,[df_V])
    ]

    list_nom_df = [  
            "No Added Datas",
            "FPTV" ,
            "F" ,
            "FP" ,
            "FT",
            "FV" ,
            "FPT" ,
            "FPV" ,
            "FTV" ,
            "P" ,
            "PT" ,
            "PV" ,
            "PTV" ,
            "T" ,
            "TV" ,
            "V" 
    ]



    return list_df, list_nom_df

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

def build_df_datas_choice(list_nom_df, list_df, secteur):
    results = pd.DataFrame(columns=['Nom', 'Train_score', 'Test_score', 'Ecart'])
    for nom_df, df,i in zip(list_nom_df,list_df, stqdm(range(16))) : 

        gridcv_GRB, X_train_scaled, X_test_scaled, y_train, y_test =  train_model(df,GBR,params_gbr,secteur)  
        Train_score = gridcv_GRB.score(X_train_scaled, y_train)
        Test_score = gridcv_GRB.score(X_test_scaled, y_test)
        results = results.append(
                        {
                            'Nom' : nom_df,
                            'Train_score' : Train_score,
                            'Test_score' : Test_score,
                            'Ecart' : Train_score - Test_score                      
                            
                        } , ignore_index=True
        
        )

    df_datas_choice = results.sort_values(['Test_score', 'Ecart'],
              ascending = [False, True])
    
    return df_datas_choice


def concat_df_test (df_min,list_df):    
    for df_ in list_df:
          df_min = pd.concat([df_min,df_], axis=1)
    return df_min





def train_model(df, model, param, secteur) : 
    # st.write(df)
    target = df[secteur]
    features = df.drop(secteur, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=22)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    X_test_scaled = scaler.transform(X_test)  
    
    gridcv =  GridSearchCV(estimator = model, param_grid = param)
    gridcv.fit(X_train_scaled, y_train)
    
    return gridcv, X_train_scaled, X_test_scaled, y_train, y_test


def store_best_datas(secteur, score_test, ecart, added_datas, data_selection):
    df = pd.read_csv('datas/df_secteur_best_datas.csv')
    secteur_serie = df[df['secteur']==secteur]
    score_test_stored = secteur_serie.iloc[0,1]
    ecart_stored = secteur_serie.iloc[0,2]
    basis_data = secteur_serie.iloc[0,3]

    if (score_test > score_test_stored  and ecart < ecart_stored) or (score_test == score_test_stored  and ecart == ecart_stored and len(data_selection)< len(basis_data)): 
        df.drop( df[ df['secteur'] == secteur ].index, inplace=True)
        df = df.append({
                    'secteur': secteur,
                    'score': score_test,    
                    'ecart': ecart, 
                    'basis_data': data_selection,
                    'added_datas': added_datas      
        
                    }, ignore_index=True)
        df.to_csv('datas/df_secteur_best_datas.csv', index=False)

    return df
     



def construc_best_dataset_secteur(secteur, df_min, df_F, df_P, df_V, df_T) :
    df = pd.read_csv('datas/df_secteur_best_datas.csv')
    secteur_serie = df[df['secteur']==secteur]
    basis_data = secteur_serie.iloc[0,3]
    added_datas = secteur_serie.iloc[0,4]
    df_secteur = drop_datas(df_min,basis_data)

    suppr=[]
    if secteur == 'REALISE_TOTAL_FRAIS':
        suppr.append('REALISE_TOTAL_GEL')
        suppr.append('REALISE_TOTAL_FFL')
    elif secteur == 'REALISE_TOTAL_GEL':
        suppr.append('REALISE_TOTAL_FRAIS')
        suppr.append('REALISE_TOTAL_FFL')
    elif secteur == 'REALISE_TOTAL_FFL':
        suppr.append('REALISE_TOTAL_FRAIS')
        suppr.append('REALISE_TOTAL_GEL')

    df_secteur = df_secteur.drop(suppr, axis = 1)

    if 'F' in  added_datas:
        df_secteur = pd.concat([df_secteur, df_F], axis=1)

    if 'P' in  added_datas:
        df_secteur = pd.concat([df_secteur, df_P], axis=1)

    if 'T' in  added_datas:
        df_secteur = pd.concat([df_secteur, df_T], axis=1)

    if 'V' in  added_datas:
        df_secteur = pd.concat([df_secteur, df_V], axis=1)

    return df_secteur 


def get_mae_per_day(y_test_df_merge, y_test_ri, y_pred_array, secteur) :
    from sklearn.metrics import mean_absolute_error
    # retrouver les index des lignes y_test dans les pred 
    pred=[]
    for index, data in zip(y_test_ri['index'], y_pred_array):
        pred.append([index,data])
        pred_pd = pd.DataFrame(pred, columns=['index','y_pred'])
     
    
    # réunir dans un même df les test et les pred
    y_test_pred_merge=pred_pd.merge(y_test_df_merge, how='left', on='index')
#     print(y_test_pred_merge)
    # En utilisant le rapprochement test / pred / weekday, calculer la mae par weekday
    mean_per_weekday=[0,0,0,0,0,0]        
    for i in range(6):
        y_test_pred_merge_i = y_test_pred_merge[y_test_pred_merge['weekday']==i]
        y_test_i = y_test_pred_merge_i[secteur]
        y_pred_i = y_test_pred_merge_i['y_pred']
        mean_per_weekday[i]=mean_absolute_error(y_test_i, y_pred_i)
        
    return mean_per_weekday 


def get_mse_per_day(y_test_df_merge, y_test_ri, y_pred_array, secteur) :
    from sklearn.metrics import mean_squared_error
    # retrouver les index des lignes y_test dans les pred 
    pred=[]
    for index, data in zip(y_test_ri['index'], y_pred_array):
        pred.append([index,data])
        pred_pd = pd.DataFrame(pred, columns=['index','y_pred'])
     
    
    # réunir dans un même df les test et les pred
    y_test_pred_merge=pred_pd.merge(y_test_df_merge, how='left', on='index')
#     print(y_test_pred_merge)
    # En utilisant le rapprochement test / pred / weekday, calculer la mae par weekday
    mean_per_weekday=[0,0,0,0,0,0]        
    for i in range(6):
        y_test_pred_merge_i = y_test_pred_merge[y_test_pred_merge['weekday']==i]
        y_test_i = y_test_pred_merge_i[secteur]
        y_pred_i = y_test_pred_merge_i['y_pred']
        mean_per_weekday[i]=mean_squared_error(y_test_i, y_pred_i)
        
    return mean_per_weekday 


           
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
    return df

def drop_time_and_index_fields(df):
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


 
def numOfDays(date1, date2):
    return (date2-date1).days
     


def add_time_datas(date_debut,date_fin,df):
    nb_days = numOfDays(date_debut, date_fin)
    dates_range=[date_debut]
    temp_date =date_debut
    for i in range(nb_days):
        new_date = temp_date + timedelta(days=1)
        temp_date=new_date
        dates_range.append(new_date)
        i+=1

    df['DATE'] = dates_range
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

def add_vacances(df_origin):

    df = pd.read_csv("datas/vacancesFrance.csv", sep =',')
    df['date'] = pd.to_datetime(df['date'], errors='ignore')
    df.sort_values(by='date', inplace=True) #import to sort the df by date
    df.reset_index(drop=True)

    #Filtre sur la période qui nous intéresse

    df = df[(df['date'] > '2019-12-31') & (df['date'] < '2023-06-01')]
    df.sort_values(by='date', inplace=True) #import to sort the df by date
    df = df.replace({True:1, False:0})
    df = df.rename(columns={"date": "DATE"})

    # Renomme les vacances
    df = df.replace(

        { 
            "Pont de l'Ascension":"Ascen",
            "Vacances d'hiver":"winter",
            "Vacances d'été":"summer",
            "Vacances de Noël":"xmas",
            "Vacances de la Toussaint":"halloween",
            "Vacances de printemps":"spring"
        }

    )

    df_holydays = df[(df['nom_vacances'].isna()==False)]
    df = pd.get_dummies(df, columns=['nom_vacances'])
    
    df_origin['DATE'] = pd.to_datetime(df_origin['DATE'], errors='ignore')
    df_utlime =  df_origin.merge(df, on='DATE', how='left')

    df_utlime['to_holydays_a']= df_utlime['DATE'].apply(lambda date: nb_days_to_next_holyday (date, df_holydays,"a"))
    df_utlime['to_holydays_b']= df_utlime['DATE'].apply(lambda date: nb_days_to_next_holyday (date, df_holydays,"b"))
    df_utlime['to_holydays_c']= df_utlime['DATE'].apply(lambda date: nb_days_to_next_holyday (date, df_holydays,"c"))

    df_utlime['from_holydays_a']= df_utlime['DATE'].apply(lambda date: nb_days_from_next_holyday (date, df_holydays,"a"))
    df_utlime['from_holydays_b']= df_utlime['DATE'].apply(lambda date: nb_days_from_next_holyday (date, df_holydays,"b"))
    df_utlime['from_holydays_c']= df_utlime['DATE'].apply(lambda date: nb_days_from_next_holyday (date, df_holydays,"c"))

    return df_utlime


def nb_days_to_next_holyday (date, df_holydays, zone): 
    zone = "vacances_zone_" + zone
#     print(zone)
    df_holydays = df_holydays[(df_holydays['DATE']>=date) & (df_holydays[zone]==1)].sort_values(by='DATE',ascending=True)
    date_next_holidays = df_holydays.iloc[0,0:1]['DATE']
    return (date_next_holidays - date).days
    
def nb_days_from_next_holyday (date, df_holydays, zone): 
    zone = "vacances_zone_" + zone
    df_holydays = df_holydays[(df_holydays['DATE']<=date) & (df_holydays[zone]==1)].sort_values(by='DATE',ascending=False)
    date_next_holidays = df_holydays.iloc[0,0:1]['DATE']
    return ( date - date_next_holidays).days

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



    