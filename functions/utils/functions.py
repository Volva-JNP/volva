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

    
   
