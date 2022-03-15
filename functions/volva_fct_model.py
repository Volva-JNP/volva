from this import d
import pandas as pd
import numpy as np
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
from calendar import monthrange
from functions.utils.functions import *


from sklearn.model_selection import train_test_split




def build_page_model():

    df, df_total, df_minimum = get_df()
    st.session_state.list_ecart_mae = []
    st.session_state.list_ecart_mse = []
    st.session_state.list_nom_model = []

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    with st.expander('Ajouter des données'):
        uploaded_file = st.file_uploader("Choisissez un fichier")
        if uploaded_file is not None:
            uploaded = pd.read_excel(uploaded_file, header=0)
            

            if len(uploaded.columns) == 4 and \
                "DATE" in uploaded.columns  and \
                "REALISE_TOTAL_FRAIS" in uploaded.columns  and \
                "REALISE_TOTAL_GEL" in uploaded.columns  and \
                "REALISE_TOTAL_FFL" in uploaded.columns  :

                uploaded_year = uploaded.loc[0,'DATE'].year
                uploaded_month = uploaded.loc[0,'DATE'].month
                num_days = monthrange(uploaded_year, uploaded_month)[1]
                nb_line = uploaded.shape[0]
                if   nb_line != num_days  :
                    st.warning("Le nombre de lignes ne correspond pas au nombre de jours dans le mois.")  
                else : 
                    placeholder = st.empty()
                    placeholder.warning("Le dataset est en cours de préparation. Veuillez patienter ...")
                    df = uploaded
                    date_debut = df.loc[0,'DATE']
                    date_fin = df.loc[nb_line-1,'DATE']
                    df = add_time_datas(date_debut,date_fin)
                    df = add_holydays(df)
                    df = add_promotions(df)
                    df = add_temperatures_data(df)            
                    df = drop_time_and_index_fields(df)
                    st.write(df)
                    placeholder = st.empty()

            else:
                
                st.warning("Le format de fichier ne correspond pas à celui attendu.  \n" \
                            " Il doit contenir 4 colonnes :  \n"\
                            "- DATE  \n"\
                            "- REALISE_TOTAL_FRAIS  \n"\
                            "- REALISE_TOTAL_GEL  \n"\
                            "- REALISE_TOTAL_FFL  \n"\
                            + str(len(uploaded.columns))                            
                )
                
    

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

        st.session_state.list_ecart_mae = []
        st.session_state.list_nom_model = []
        st.session_state.list_ecart_mse = []

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

        st.session_state.list_ecart_mae = []
        st.session_state.list_nom_model = []
        st.session_state.list_ecart_mse = []

        secteur = 'REALISE_TOTAL_GEL'
        df_FPTV, df_min, df_F, df_P, df_V, df_T = build_df(df,'REALISE_TOTAL_GEL', data_selection)        
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

        st.session_state.list_ecart_mae = []
        st.session_state.list_nom_model = []
        st.session_state.list_ecart_mse = []

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
            st.write(modelregression)
        with col2:
            st.write("")
            st.image("img/MLregression.jpg")

       
    if menu_secteur != 'vide':  
        
        df_kept = construc_best_dataset_secteur(secteur, df_minimum, df_F, df_P, df_V, df_T)
        with st.expander('Voir le dataset'):   
            st.write(df_kept)
        
        st.write("")
        st.write("")

        with st.expander("GradientBoostingRegressor"):               
            st.subheader("Test d'un modèle GradientBoostingRegressor")
            model_GBR, y_test, pred_test_GBR = test_model(df_kept,GBR,params_gbr,secteur, "GBR")
            if 'model_GBR_'+ secteur in st.session_state:            
                display_test_pred_graph(y_test, pred_test_GBR , df_total)
                if 'y_test_df_merge' not in st.session_state:
                    y_test_ri = y_test.reset_index()
                    df_ri = df_total.reset_index()
                    y_test_df_merge = y_test_ri.merge(df_ri[['index', 'weekday']], how='left', on='index')
                else :
                    y_test_df_merge = st.session_state.y_test_df_merge

                mae_per_day_GBR = get_mae_per_day(y_test_df_merge,y_test_ri,pred_test_GBR,secteur)
                mse_per_day_GBR = np.sqrt(get_mse_per_day(y_test_df_merge,y_test_ri,pred_test_GBR, secteur))

                st.session_state.list_ecart_mae.append(mae_per_day_GBR)
                st.session_state.list_ecart_mse.append(mse_per_day_GBR)
                st.session_state.list_nom_model.append("GBR")



        with st.expander("BayesianRidge"):  
            from sklearn.linear_model import BayesianRidge
            BR = BayesianRidge()
            params_BR = { }
            st.subheader("Test d'un modèle BayesianRidge")
            model_BR, y_test, pred_test_BR = test_model(df_kept,BR,params_BR,secteur,"BR") 
            if 'model_BR_'+ secteur in st.session_state:              
                display_test_pred_graph(y_test, pred_test_BR , df_total)
                if 'y_test_df_merge' not in st.session_state:
                    y_test_ri = y_test.reset_index()
                    df_ri = df_total.reset_index()
                    y_test_df_merge = y_test_ri.merge(df_ri[['index', 'weekday']], how='left', on='index')
                else :
                    y_test_df_merge = st.session_state.y_test_df_merge
                
                mae_per_day_ADABR = get_mae_per_day(y_test_df_merge,y_test_ri,pred_test_BR, secteur)
                mse_per_day_ADABR = np.sqrt(get_mse_per_day(y_test_df_merge,y_test_ri,pred_test_BR, secteur))

                st.session_state.list_ecart_mae.append(mae_per_day_ADABR)
                st.session_state.list_ecart_mse.append(mse_per_day_ADABR)
                st.session_state.list_nom_model.append("BR")


        with st.expander("Lasso"):  
            from sklearn.linear_model import Lasso 
            lasso = Lasso()
            params_lasso = { }
            st.subheader("Test d'un modèle Lasso")
            model_lasso, y_test, pred_test_lasso = test_model(df_kept,lasso,params_lasso,secteur,"LASSO") 
            if 'model_LASSO_'+ secteur in st.session_state:              
                display_test_pred_graph(y_test, pred_test_lasso , df_total)
                if 'y_test_df_merge' not in st.session_state:
                    y_test_ri = y_test.reset_index()
                    df_ri = df_total.reset_index()
                    y_test_df_merge = y_test_ri.merge(df_ri[['index', 'weekday']], how='left', on='index')
                else :
                    y_test_df_merge = st.session_state.y_test_df_merge   

                mae_per_day_lasso = get_mae_per_day(y_test_df_merge,y_test_ri,pred_test_lasso, secteur)
                mse_per_day_lasso = np.sqrt(get_mse_per_day(y_test_df_merge,y_test_ri,pred_test_lasso, secteur))  

                st.session_state.list_ecart_mae.append(mae_per_day_lasso)
                st.session_state.list_ecart_mse.append(mse_per_day_lasso)
                st.session_state.list_nom_model.append("LASSO")


        with st.expander("RandomForestRegressor"):  
            from sklearn.ensemble import RandomForestRegressor
            rfr = RandomForestRegressor( )
            params_rfr = {
                        "max_depth":[9,10,11], 
                        "n_estimators":[2000]
                            }
            st.subheader("Test d'un modèle RandomForestRegressor")
            model_rfr, y_test, pred_test_rfr  = test_model(df_kept,rfr,params_rfr,secteur,"RFR") 
            if 'model_RFR_'+ secteur in st.session_state:              
                display_test_pred_graph(y_test, pred_test_rfr , df_total) 
                if 'y_test_df_merge' not in st.session_state:
                    y_test_ri = y_test.reset_index()
                    df_ri = df_total.reset_index()
                    y_test_df_merge = y_test_ri.merge(df_ri[['index', 'weekday']], how='left', on='index')
                else :
                    y_test_df_merge = st.session_state.y_test_df_merge  
                
                mae_per_day_rfr = get_mae_per_day(y_test_df_merge,y_test_ri,pred_test_rfr, secteur)
                mse_per_day_rfr = np.sqrt(get_mse_per_day(y_test_df_merge,y_test_ri,pred_test_rfr, secteur))

                st.session_state.list_ecart_mae.append(mae_per_day_rfr)
                st.session_state.list_ecart_mse.append(mse_per_day_rfr)
                st.session_state.list_nom_model.append("RFR")



        with st.expander("KNeighborsRegressor"): 
            from sklearn.neighbors import KNeighborsRegressor
            KNR= KNeighborsRegressor()
            params_KNR = {
                        "n_neighbors":[3,4,5,6,7] 
                        }
            st.subheader("Test d'un modèle KNeighborsRegressor")
            model_KNR,  y_test, pred_test_KNR = test_model(df_kept,KNR,params_KNR,secteur,"KNR")
            if 'model_KNR_'+ secteur in st.session_state:              
                display_test_pred_graph(y_test, pred_test_KNR , df_total)
                if 'y_test_df_merge' not in st.session_state:
                    y_test_ri = y_test.reset_index()
                    df_ri = df_total.reset_index()
                    y_test_df_merge = y_test_ri.merge(df_ri[['index', 'weekday']], how='left', on='index')
                else :
                    y_test_df_merge = st.session_state.y_test_df_merge   

                mae_per_day_KNR = get_mae_per_day(y_test_df_merge,y_test_ri,pred_test_KNR, secteur)
                mse_per_day_KNR = np.sqrt(get_mse_per_day(y_test_df_merge,y_test_ri,pred_test_KNR, secteur))

                st.session_state.list_ecart_mae.append(mae_per_day_KNR)
                st.session_state.list_ecart_mse.append(mse_per_day_KNR)
                st.session_state.list_nom_model.append("KNR")


        
        with st.expander("ElasticNetCV"): 
            from sklearn.linear_model import ElasticNetCV 
            EN = ElasticNetCV(l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0))
            params_EN= {
             }
            st.subheader("Test d'un modèle ElasticNetCV")
            model_EN , y_test, pred_test_EN = test_model(df_kept,EN,params_EN,secteur,"EN")
            if 'model_EN_'+ secteur in st.session_state:              
                display_test_pred_graph(y_test, pred_test_EN , df_total)
                if 'y_test_df_merge' not in st.session_state:
                    y_test_ri = y_test.reset_index()
                    df_ri = df_total.reset_index()
                    y_test_df_merge = y_test_ri.merge(df_ri[['index', 'weekday']], how='left', on='index')
                else :
                    y_test_df_merge = st.session_state.y_test_df_merge  
                
                mae_per_day_EN = get_mae_per_day(y_test_df_merge,y_test_ri,pred_test_EN, secteur)
                mse_per_day_EN = np.sqrt(get_mse_per_day(y_test_df_merge,y_test_ri,pred_test_EN, secteur))

                st.session_state.list_ecart_mae.append(mae_per_day_EN)
                st.session_state.list_ecart_mse.append(mse_per_day_EN)
                st.session_state.list_nom_model.append("EN")



        with st.expander("DecisionTreeRegressor"): 
            from sklearn.tree import DecisionTreeRegressor
            DTR = DecisionTreeRegressor()
            params_DTR= {
                            'max_depth':[4,5,6,7,8]
                        }
            st.subheader("Test d'un modèle DecisionTreeRegressor")
            model_DTR,  y_test, pred_test_DTR = test_model(df_kept,DTR,params_DTR,secteur,"DTR") 
            if 'model_DTR_'+ secteur in st.session_state:              
                display_test_pred_graph(y_test, pred_test_DTR , df_total)
                if 'y_test_df_merge' not in st.session_state:
                    y_test_ri = y_test.reset_index()
                    df_ri = df_total.reset_index()
                    y_test_df_merge = y_test_ri.merge(df_ri[['index', 'weekday']], how='left', on='index')
                else :
                    y_test_df_merge = st.session_state.y_test_df_merge

                mae_per_day_DTR = get_mae_per_day(y_test_df_merge,y_test_ri,pred_test_DTR, secteur)
                mse_per_day_DTR = np.sqrt(get_mse_per_day(y_test_df_merge,y_test_ri,pred_test_DTR, secteur))

                st.session_state.list_ecart_mae.append(mae_per_day_DTR)
                st.session_state.list_ecart_mse.append(mse_per_day_DTR)
                st.session_state.list_nom_model.append("DTR")




    st.title('Comparaison des modèles')
    with st.expander('Information'):    
        col1, col2 = st.columns(2)
        with col1:
            st.write(comparaison)
        with col2:
            st.image("img/machine_learning_modele_.png")
    # st.write(st.session_state.list_ecart_mae)

    if len(st.session_state.list_nom_model)>0:
        df_ecart_MAE = display_model_comparaison(st.session_state.list_ecart_mae, st.session_state.list_nom_model, "MAE")
        df_ecart_RMSE = display_model_comparaison(st.session_state.list_ecart_mse, st.session_state.list_nom_model, "MSE")

        st.title('Construction du modèle final')

        df_ecart = pd.concat([df_ecart_MAE, df_ecart_RMSE])
        df_ecart_mean = df_ecart.groupby(['jour','model']).agg({'ecart':'mean'})        
        df_ecart_mean = df_ecart_mean.reset_index() 

        lun_results =  df_ecart_mean[(df_ecart_mean['jour'] == 'Lun') ]
        min_lun_results = lun_results['ecart'].idxmin()

        mar_results =  df_ecart_mean[(df_ecart_mean['jour'] == 'Mar') ]
        min_mar_results = mar_results['ecart'].idxmin()

        mer_results =  df_ecart_mean[(df_ecart_mean['jour'] == 'Mer') ]
        min_mer_results = mer_results['ecart'].idxmin()

        jeu_results =  df_ecart_mean[(df_ecart_mean['jour'] == 'Jeu') ]
        min_jeu_results = jeu_results['ecart'].idxmin()

        ven_results =  df_ecart_mean[(df_ecart_mean['jour'] == 'Ven') ]
        min_ven_results = ven_results['ecart'].idxmin()

        sam_results =  df_ecart_mean[(df_ecart_mean['jour'] == 'Sam') ]
        min_sam_results = sam_results['ecart'].idxmin()

      
        jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven','Sam']
        best_models = [

                lun_results.loc[min_lun_results,'model'],
                mar_results.loc[min_mar_results,'model'],
                mer_results.loc[min_mer_results,'model'],
                jeu_results.loc[min_jeu_results,'model'],
                ven_results.loc[min_ven_results,'model'],
                sam_results.loc[min_sam_results,'model'],

        ]
 
        df_best_model_per_day = pd.DataFrame()
        df_best_model_per_day['Jours'] = jours
        df_best_model_per_day['Modèle'] = best_models

        
        st.write(df_best_model_per_day)

        button_keep_model = st.button("Enregistre le modèle")

        if button_keep_model:
            df_best_model_per_day.to_csv('datas/' + 'df_best_model_per_day-' + secteur + '.csv', index=False)
            for model_name in df_best_model_per_day['Modèle'].unique():
                # st.write(model)
                if model_name == "GBR":
                    model = model_GBR
                elif model_name == "BR":
                    model = model_BR
                elif model_name == "LASSO":
                    model = model_lasso
                elif model_name == "RFR":
                    model = model_rfr
                elif model_name == "KNR":
                    model = model_KNR
                elif model_name == "EN":
                    model = model_EN
                elif model_name == "DTR":
                    model = model_DTR

                dump(model, 'models/model_' + model_name +'_' + secteur + '.joblib')






