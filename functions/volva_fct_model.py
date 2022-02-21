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

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, train_test_split , cross_val_score, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import MinMaxScaler 
from  sklearn.linear_model import LogisticRegression 
from  sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from math import *
from sklearn.ensemble import GradientBoostingRegressor
from stqdm import stqdm

import seaborn as sns

from sklearn.model_selection import train_test_split

path = 'datas/volva_datas_utlimate_one.csv'


GBR = GradientBoostingRegressor() # Params pour GEL
params_gbr = {    
        'max_depth': [1], 
#         'n_estimators': [i for i in np.arange(1100, 1500,100 )],
#         'learning_rate': [i for i in np.arange(0.0095, 0.010,0.0001 )]
        'n_estimators': [1400],
        'learning_rate': [0.099]
}




def build_page_model():

    st.title('Selection des données utiles par test de modèles')
    st.write('Sélectionner un secteur:')
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if 'secteur' not in st.session_state:
        st.session_state.secteur = 'REALISE_TOTAL_FRAIS'
        
    menu = st.radio(    
    "",
    ("secteur frais", "secteur Gel", "secteur FFL"),
    )

    if menu =='secteur frais':    

        secteur = 'REALISE_TOTAL_FRAIS'
        st.write('Analyse du secteur ', secteur, ' en cours ...')
        df_FPTV, df_min, df_F, df_P, df_V, df_T = build_df('REALISE_TOTAL_FRAIS')
        st.write("2")
        list_df, list_nom_df = build_list_test(df_FPTV, df_min, df_F, df_P, df_V, df_T)
        st.write("3")
        df_datas_choice = build_df_datas_choice(list_nom_df, list_df, secteur)
        st.write("4")

    

    if menu == 'secteur Gel':  

        secteur = 'REALISE_TOTAL_GEL'
        st.write('Analyse du secteur ', secteur, ' en cours ...')
        df_FPTV, df_min, df_F, df_P, df_V, df_T = build_df('REALISE_TOTAL_GEL')
        list_df, list_nom_df = build_list_test(df_FPTV, df_min, df_F, df_P, df_V, df_T)
        df_datas_choice = build_df_datas_choice(list_nom_df, list_df, secteur)

        
        
    if menu == 'secteur FFL':  

        secteur = 'REALISE_TOTAL_FFL'  
        st.write('Analyse du secteur ', secteur, ' en cours ...')
        df_FPTV, df_min, df_F, df_P, df_V, df_T = build_df('REALISE_TOTAL_FFL')
        list_df, list_nom_df = build_list_test(df_FPTV, df_min, df_F, df_P, df_V, df_T)
        df_datas_choice = build_df_datas_choice(list_nom_df, list_df,secteur)


    st.write(df_datas_choice)



def build_df(secteur):
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

    if secteur == 'REALISE_TOTAL_FRAIS':
        suppr.append('REALISE_TOTAL_GEL')
        suppr.append('REALISE_TOTAL_FFL')
    elif secteur == 'REALISE_TOTAL_GEL':
        suppr.append('REALISE_TOTAL_FRAIS')
        suppr.append('REALISE_TOTAL_FFL')
    elif secteur == 'REALISE_TOTAL_FFL':
        suppr.append('REALISE_TOTAL_FRAIS')
        suppr.append('REALISE_TOTAL_GEL')



    df_clean = df.drop(suppr, axis = 1)

    df_FPTV = pd.concat([df_clean.iloc[:, :44],df_clean.iloc[:, 46:]], axis=1)
    df_min = pd.concat([df_clean.iloc[:, :3],df_clean.iloc[:, 38:44]], axis=1)

    df_F = df_clean.iloc[:, 3:5]
    df_P = df_clean.iloc[:, 5:38]
    df_T = df_clean.iloc[:, 46:48]
    df_V = df_clean.iloc[:, 48:]

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

    
def build_df_datas_choice(list_nom_df, list_df, secteur):
    st.write('3.1')
    
    results = pd.DataFrame(columns=['Nom', 'Train_score', 'Test_score', 'Ecart'])
    st.write('3.2')
    for nom_df, df,i in zip(list_nom_df,list_df, stqdm(range(16))) : 
    # for nom_df, df in zip(list_nom_df,list_df) :   
        st.write('3.3')
        gridcv_GRB, X_train_scaled, X_test_scaled, y_train, y_test =  train_model(df,GBR,params_gbr,secteur)  
        st.write('3.4')
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
    
    target = df[secteur]
    features = df.drop(secteur, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=22)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    X_test_scaled = scaler.transform(X_test)  
    
    gridcv =  GridSearchCV(estimator = model, param_grid = param)
    gridcv.fit(X_train_scaled, y_train)
    
    return gridcv, X_train_scaled, X_test_scaled, y_train, y_test


# def get_mae_per_day(y_test_df_merge, y_test_ri, y_pred_array) :
#     from sklearn.metrics import mean_absolute_error
#     # retrouver les index des lignes y_test dans les pred 
#     pred=[]
#     for index, data in zip(y_test_ri['index'], y_pred_array):
#         pred.append([index,data])
#         pred_pd = pd.DataFrame(pred, columns=['index','y_pred'])
     
    
#     # réunir dans un même df les test et les pred
#     y_test_pred_merge=pred_pd.merge(y_test_df_merge, how='left', on='index')
# #     print(y_test_pred_merge)
#     # En utilisant le rapprochement test / pred / weekday, calculer la mae par weekday
#     mean_per_weekday=[0,0,0,0,0,0]        
#     for i in range(6):
#         y_test_pred_merge_i = y_test_pred_merge[y_test_pred_merge['weekday']==i]
#         y_test_i = y_test_pred_merge_i[secteur]
#         y_pred_i = y_test_pred_merge_i['y_pred']
#         mean_per_weekday[i]=mean_absolute_error(y_test_i, y_pred_i)
        
#     return mean_per_weekday 


# def get_mse_per_day(y_test_df_merge, y_test_ri, y_pred_array) :
#     from sklearn.metrics import mean_squared_error
#     # retrouver les index des lignes y_test dans les pred 
#     pred=[]
#     for index, data in zip(y_test_ri['index'], y_pred_array):
#         pred.append([index,data])
#         pred_pd = pd.DataFrame(pred, columns=['index','y_pred'])
     
    
#     # réunir dans un même df les test et les pred
#     y_test_pred_merge=pred_pd.merge(y_test_df_merge, how='left', on='index')
# #     print(y_test_pred_merge)
#     # En utilisant le rapprochement test / pred / weekday, calculer la mae par weekday
#     mean_per_weekday=[0,0,0,0,0,0]        
#     for i in range(6):
#         y_test_pred_merge_i = y_test_pred_merge[y_test_pred_merge['weekday']==i]
#         y_test_i = y_test_pred_merge_i[secteur]
#         y_pred_i = y_test_pred_merge_i['y_pred']
#         mean_per_weekday[i]=mean_squared_error(y_test_i, y_pred_i)
        
#     return mean_per_weekday 



    
