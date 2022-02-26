from this import d
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




GBR = GradientBoostingRegressor() # Params pour GEL
params_gbr = {    
        'max_depth': [1], 
#         'n_estimators': [i for i in np.arange(1100, 1500,100 )],
#         'learning_rate': [i for i in np.arange(0.0095, 0.010,0.0001 )]
        'n_estimators': [1400],
        'learning_rate': [0.099]
}


def build_page_model():

    df, df_total, df_minimum = get_df()

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


    st.title('Selection des données utiles par test de modèles')
    with st.expander('Information'):    
        col1, col2 = st.columns(2)
        with col1:
            st.write("A compléter")
        with col2:
            st.write("A compléter")


    st.write("Sélectionner les données à laisser dans le dataset d'origine")

    col3, col4, col5 = st.columns(3)
    with col3:   
        year_check= st.checkbox('Année', value=True)
        month_period_check = st.checkbox('Période du mois', value=True)

    with col4:
        month_day_check = st.checkbox('Jour du Mois', value=True)
        week_day_check = st.checkbox('Jour de la Semaine', value=True)

    with col5:
        month_check = st.checkbox('Mois', value=True)   
        week_check = st.checkbox('Semaine', value=True)

    data_selection=""
    if year_check and "A_" not in data_selection:
        data_selection = data_selection + "A_"
    if month_period_check and "MP_" not in data_selection:
        data_selection = data_selection + "MP_"
    if month_day_check and "MD_" not in data_selection:
        data_selection = data_selection + "MD_"
    if week_day_check and "WD_" not in data_selection:
        data_selection = data_selection + "WD_"
    if month_check and "M_" not in data_selection:
        data_selection = data_selection + "M_"
    if week_check and "W_" not in data_selection:
        data_selection = data_selection + "W_"

    st.write("")
    st.write("")
    st.write("Sélectionner le secteur")

    menu_secteur  = st.radio(    
    "",
    ("vide", "secteur frais", "secteur GEL", "secteur FFL"),

    )
  

    if menu_secteur =='secteur frais':   

        secteur = 'REALISE_TOTAL_FRAIS'
        df_FPTV, df_min, df_F, df_P, df_V, df_T = build_df(df,'REALISE_TOTAL_FRAIS', data_selection)    
        try:
             df_datas_choice = pd.read_csv('datas/df_datas_choice_' + data_selection + secteur + '.csv')
             df_datas_choice = df_datas_choice[["Nom","Train_score","Test_score","Ecart"]]

        except FileNotFoundError as fnfe:
            placeholder = st.empty()
            placeholder.warning("Cette hypothèse n'a pas encore été testée. Veuillez patienter pendant son évaluation ...")
            list_df, list_nom_df = build_list_test(df_FPTV, df_min, df_F, df_P, df_V, df_T)
            df_datas_choice = build_df_datas_choice(list_nom_df, list_df, secteur)
            df_datas_choice.to_csv('datas/df_datas_choice_' + data_selection + secteur + '.csv',  index=False)
            placeholder.empty()

        

    if menu_secteur == 'secteur GEL':  

        secteur = 'REALISE_TOTAL_GEL'
        df_FPTV, df_min, df_F, df_P, df_V, df_T = build_df(df,'REALISE_TOTAL_FFL', data_selection)        
        try:
            df_datas_choice = pd.read_csv('datas/df_datas_choice_' + data_selection + secteur + '.csv')
            df_datas_choice = df_datas_choice[["Nom","Train_score","Test_score","Ecart"]]

        except FileNotFoundError as fnfe:
            placeholder = st.empty()
            placeholder.warning("Cette hypothèse n'a pas encore été testée. Veuillez patienter pendant son évaluation ...")
            list_df, list_nom_df = build_list_test(df_FPTV, df_min, df_F, df_P, df_V, df_T)
            df_datas_choice = build_df_datas_choice(list_nom_df, list_df, secteur)
            df_datas_choice.to_csv('datas/df_datas_choice_' + data_selection + secteur + '.csv', index=False)
            placeholder.empty()
        
        
    if menu_secteur == 'secteur FFL':  

        secteur = 'REALISE_TOTAL_FFL' 
        df_FPTV, df_min, df_F, df_P, df_V, df_T = build_df(df,'REALISE_TOTAL_FFL', data_selection)
        try:
            df_datas_choice = pd.read_csv('datas/df_datas_choice_' + data_selection + secteur + '.csv')
            df_datas_choice = df_datas_choice[["Nom","Train_score","Test_score","Ecart"]]

        except FileNotFoundError as fnfe:
            placeholder = st.empty()
            placeholder.warning("Cette hypothèse n'a pas encore été testée. Veuillez patienter pendant son évaluation ...")             
            list_df, list_nom_df = build_list_test(df_FPTV, df_min, df_F, df_P, df_V, df_T)
            df_datas_choice = build_df_datas_choice(list_nom_df, list_df,secteur)
            df_datas_choice.to_csv('datas/df_datas_choice_' + data_selection + secteur + '.csv', index=False)
            placeholder.empty()
            

    if menu_secteur != 'vide':  
        st.caption("Meilleur résultat pour le " + menu_secteur)
        st.write(df_datas_choice.iloc[0:1])  
        with st.expander('Voir le résultats de tous les tests'):
            st.write(df_datas_choice)    
        score_test = df_datas_choice.iloc[0,2]
        ecart = df_datas_choice.iloc[0,3]
        added_datas = df_datas_choice.iloc[0,0]
        with st.expander("Voir le meilleur paramètrage pour chaque secteur"): 
            df_best_results = store_best_datas(secteur, score_test, ecart, added_datas,data_selection)
            st.write(df_best_results)


    st.title('Tests des modèles de regression')
    with st.expander('Information'):    
        col1, col2 = st.columns(2)
        with col1:
            st.write("A compléter")
        with col2:
            st.write("A compléter")

       
    if menu_secteur != 'vide':  
        
        df_kept = construc_best_dataset_secteur(secteur, df_minimum, df_F, df_P, df_V, df_T)
        with st.expander('Voir le dataset'):   
            st.write(df_kept)
        
        st.write("")
        st.write("")

        with st.expander("GradientBoostingRegressor"):   
            st.subheader("Test d'un modèle GradientBoostingRegressor")
            test_model(df_kept,GBR,params_gbr,secteur, "GBR", df_total)

        with st.expander("BayesianRidge"):  
            from sklearn.linear_model import BayesianRidge
            BR = BayesianRidge()
            params_BR = { }
            st.subheader("Test d'un modèle BayesianRidge")
            test_model(df_kept,BR,params_BR,secteur,"BR", df_total) 

        with st.expander("Lasso"):  
            from sklearn.linear_model import Lasso 
            lasso = Lasso()
            params_lasso = { }
            st.subheader("Test d'un modèle Lasso")
            test_model(df_kept,lasso,params_lasso,secteur,"LASSO", df_total)      

        with st.expander("RandomForestRegressor"):  
            from sklearn.ensemble import RandomForestRegressor
            rfr = RandomForestRegressor( )
            params_rfr = {
                        "max_depth":[9,10,11], 
                        "n_estimators":[2000]
                            }
            st.subheader("Test d'un modèle RandomForestRegressor")
            test_model(df_kept,rfr,params_rfr,secteur,"RFR", df_total)    


        with st.expander("KNeighborsRegressor"): 
            from sklearn.neighbors import KNeighborsRegressor
            KNR= KNeighborsRegressor()
            params_KNR = {
                        "n_neighbors":[3,4,5,6,7] 
                        }
            st.subheader("Test d'un modèle KNeighborsRegressor")
            test_model(df_kept,KNR,params_KNR,secteur,"KNR", df_total)  

        
        with st.expander("ElasticNetCV"): 
            from sklearn.linear_model import ElasticNetCV 
            EN = ElasticNetCV(l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0))
            params_EN= {
             }
            st.subheader("Test d'un modèle ElasticNetCV")
            test_model(df_kept,EN,params_EN,secteur,"EN", df_total)  

        with st.expander("DecisionTreeRegressor"): 
            from sklearn.tree import DecisionTreeRegressor
            DTR = DecisionTreeRegressor()
            params_DTR= {
                            'max_depth':[4,5,6,7,8]
                        }
            st.subheader("Test d'un modèle DecisionTreeRegressor")
            test_model(df_kept,DTR,params_DTR,secteur,"DTR", df_total) 


            # y_test_ri = y_test.reset_index()
            # df_ri = df_total.reset_index()
            # y_test_df_merge = y_test_ri.merge(df_ri[['index', 'weekday']], how='left', on='index')
            # y_test_df_merge

            # pred_test_GBR = gridcv_GRB.predict(X_test_scaled)
            # # plt.scatter(y_test, pred_test_GBR)
            # # plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], c='r')
            # # plt.show()

            # mae_per_day_GBR = get_mae_per_day(y_test_df_merge,y_test_ri,pred_test_GBR,secteur)
            # # st.write(mae_per_day_GBR)

            # mse_per_day_GBR = np.sqrt(get_mse_per_day(y_test_df_merge,y_test_ri,pred_test_GBR,secteur))
            # # st.write(mse_per_day_GBR)
            


def test_model(df_kept,Model,params,secteur, nom_model, df_total):
    button_test_model = st.button('Test modèle ' + nom_model)
    if button_test_model:
        placeholder2 = st.empty()
        placeholder2.warning("Veuillez patienter pendant l'évaluation du modèle ...")
        gridcv_model, X_train_scaled, X_test_scaled, y_train, y_test = train_model(df_kept,Model,params,secteur)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score train",np.round(gridcv_model.score(X_train_scaled, y_train),4))
        with col2:
            st.metric("Score test",np.round(gridcv_model.score(X_test_scaled, y_test),4))   
        
        placeholder2.empty()

        pred_test = gridcv_model.predict(X_test_scaled) 

        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(x=y_test, y=pred_test,
                            marker_color = df_total['weekday'].astype('int'),
                            mode='markers',
                            name='markers'))

        fig.add_trace(go.Scatter(x=[y_test.min(),y_test.max()], y=[y_test.min(),y_test.max()],
                            mode='lines',
                            name='lines'))

        st.write(fig)
        
        # return  train_model(df_kept,Model,params,secteur)  

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
        # st.write(nom_df)
        # st.write(df.columns)
    # for nom_df, df in zip(list_nom_df,list_df) :   

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



    
