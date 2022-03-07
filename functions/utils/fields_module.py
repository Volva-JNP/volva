##### Liste des champs à conserver

fields = {
                '1':'ANNEE',
                '2':'MOIS',
                '3':'SEMAINE',
                '4':'JOUR',
                '5':'DATE',
    #             '6':'REALISE_GEL_REC',    
    #             '7':'NBHEURE_REC',
    #             '8':'OBJECTIF_PROD_REC',
    #             '9':'REALISE_FLF_REC',
    #             '10':'NBHEURE_REC',
    #             '11':'OBJECTIF_PROD_REC',
    #             '12':'REALISE_MECA_REC',
    #             '13':'NBHEURE_REC',
    #             '14':'OBJECTIF_PROD_REC',
    #             '15':'REALISE_SEC_REC',
    #             '16':'NBHEURE_REC',
    #             '17':'OBJECTIF_PROD_REC',
                '18':'PREVISION_BUDGET_FRAIS',
                '19':'REALISE_TOTAL_FRAIS',
                '20':'NB_HEURES_TOTAL_FRAIS',
                '21':'OBJECTIF_PROD_FRAIS',
                '22':'PREVISION_BUDGET_GEL',
                '23':'REALISE_TOTAL_GEL',
                '24':'NB_HEURES_TOTAL_GEL',
                '25':'OBJECTIF_PROD_GEL',
                '26':'PREVISION_BUDGET_FFL',
                '27':'REALISE_TOTAL_FFL',
                '28':'NB_HEURE_PREPARATION_FFL',
                '29':'OBJECTIF_PROD_FFL',
        #         '30':'PREVISION_BUDGET_SEC',
        #         '31':'REALISE_SEC_CPS_SEC',
        #         '32':'NBHEURE_CPS_SEC',
        #         '33':'OBJECTIF_PROD_SEC',
        #         '34':'PREVISION_BUDGET_CPA',
        #         '35':'REALISE_SEC_CPA',
        #         '36':'NBHEURE_CPA_CPA',
        #         '37':'OBJECTIF_PROD_CPA',
                '38':'REALISE_GEL_EXP',
                '39':'REALISE_FLF_EXP',
                '40':'REALISE_HGE_EXP',
                '41':'REALISE_MECA_EXP',
                '42':'TOTAL_EXPE_EXP',
                '43':'NB_HEURE_EXP',
        #         '44':'HEURES_TAQ_EXP',
                '45':'OBJECTIF_PROD_EXP'
        #         ,'46':'REALISE_SEC_EXP',
        #         '47':'NBHEURE_SEC_EXP'
 }





    
def get_fields_list():    
    fields_list = []
    for index, field in fields.items(): 
        fields_list.append(field)
    return fields_list



def get_fields():
    return fields