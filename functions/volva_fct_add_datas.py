from this import d
import pandas as pd
import numpy as np
from texts.volva_text import *
from plotly.subplots import make_subplots
from functions.volva_fct import *
from math import *

from joblib import dump
from calendar import monthrange
from functions.utils.functions import *
from datetime import date
import shutil

def build_page_add_datas():
    # redim_df()

    volva_dataset = 'volva_datas_utlimate_one.csv'
    path_datas = 'datas/'
    path_datas_archives = path_datas + 'archives/'
    

    uploaded_file = st.file_uploader("Choisissez un fichier")
    if uploaded_file is not None:
        uploaded = pd.read_excel(uploaded_file,  header=0)
        volva_datas_utlimate_one = pd.read_csv(path_datas + volva_dataset, header=0)

        ordre_colonnes = volva_datas_utlimate_one.columns

        if len(uploaded.columns) == 4   and \
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
                df = add_time_datas(date_debut,date_fin, df)
                df = add_holydays(df)
                df = add_promotions(df)
                df = add_temperatures_data(df)            
                df = add_vacances(df)
                df = drop_time_and_index_fields(df)

                df_added_datas = pd.DataFrame()

                for col in ordre_colonnes:
                    df_added_datas[col] = df[col]
                placeholder.empty()                
                # df_added_datas.to_csv('datas/added_datas.csv')

                with st.expander('Voir le dataset'):
                    volva_datas_utlimate_one['DATE']= pd.to_datetime(volva_datas_utlimate_one['DATE'], dayfirst=True) 
                    result_addition = pd.concat([volva_datas_utlimate_one,df_added_datas])
                    result_addition=result_addition.reset_index()
                    st.write(result_addition)
                    save_dataset = st.button("Sauvegarder le nouveau dataset")
                    if save_dataset:
                        today = date.today()
                        volva_dataset_path = path_datas + volva_dataset
                        volva_dataset_archives_path = path_datas_archives + str(today) + ' - ' + volva_dataset
                        shutil.copyfile(volva_dataset_path, volva_dataset_archives_path)
                        result_addition.to_csv(volva_dataset_path)


        else:
            
            st.warning("Le format de fichier ne correspond pas à celui attendu.  \n" \
                        " Il doit contenir 4 colonnes :  \n"\
                        "- DATE  \n"\
                        "- REALISE_TOTAL_FRAIS  \n"\
                        "- REALISE_TOTAL_GEL  \n"\
                        "- REALISE_TOTAL_FFL  \n"\
                        + str(uploaded.columns)                            
            )

            st.write(uploaded)
            
    
