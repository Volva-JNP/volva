
from datetime import datetime 

dict_jours_feries = {


        '2020-01-01' : 'Jour de l\’An',
        '2020-04-13' : 'Pâques',
        '2020-05-01' : 'Fête du Travail',
        '2020-05-08' : 'Victoire des Alliés en 1945',
        '2020-05-21' : 'Ascension',
        '2020-06-01' : 'Pentecôte',
        '2020-07-14' : 'Fête nationale',
        '2020-08-15' : 'Assomption',
        '2020-11-01' : 'Toussaint',
        '2020-11-11' : 'Armistice 1918',
        '2020-12-25' : 'Noël',
        '2021-01-01' : 'Jour de l’An',
        '2021-04-05' : 'Pâques',
        '2021-05-01' : 'Fête du Travail',
        '2021-05-08' : 'Victoire des Alliés en 1945',
        '2021-05-13' : 'Ascension',
        '2021-05-24' : 'Pentecôte',
        '2021-07-14' : 'Fête nationale',
        '2021-08-15' : 'Assomption',
        '2021-11-01' : 'Toussaint',
        '2021-11-11' : 'Armistice 1918',
        '2021-12-25' : 'Noël',
        '2022-01-01' : 'Jour de l\’An',
        '2022-04-18' : 'Pâques',
        '2022-05-01' : 'Fête du Travail',
        '2022-05-08' : 'Victoire des Alliés en 1945',
        '2022-05-26' : 'Ascension',
        '2022-06-06' : 'Pentecôte',
        '2022-07-14' : 'Fête nationale',
        '2022-08-15' : 'Assomption',
        '2022-11-01' : 'Toussaint',
        '2022-11-11' : 'Armistice 1918',
        '2022-12-25' : 'Noël',




}


def liste_jours_feries():
        liste_jours_feries = []
        for index, jour_feries in dict_jours_feries.items(): 
                liste_jours_feries.append(index + ":" + jour_feries )
        return liste_jours_feries


def liste_jours_feries_date():
        list_jours_feries = liste_jours_feries()
        liste_jours_feries_date = []

        for jour_ferie_nom_et_date in list_jours_feries: 
                jours_feries_split = jour_ferie_nom_et_date.split(':')
                jour_ferie=jours_feries_split[0]
                jour_ferie  = datetime.strptime(jour_ferie, "%Y-%m-%d")
                liste_jours_feries_date.append(jour_ferie)

        return liste_jours_feries_date

def get_nom_jour_ferie(jour_ferie):
        return dict_jours_feries[jour_ferie]