# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 19:03:22 2022

@author: jk
"""

#sidebar



#home

intro = '''
ANALYSE DE DONNEES D'UN SITE LOGISTIQUE 

Le projet a pour but d'analyser les historiques de volumes des différents 
services d’un site logistique pour en déduire des tendances et développer un outil prévisionnel afin d’adapter les ressources humaines nécessaires aux opérations logistiques avec une projection à plusieurs semaines.  

Nous disposions des historiques de volumes de colis jour/jour sur la période 2020/2021, du nombre d’heures travaillées sur la même période et des notions de productivité par service (18 mois d’historique).
Ces données sont issues d'une requête Oracle réalisée mensuellement.  

Nous avons donc dans un premier temps réuni l'ensemble des infos dans un seul dataset afin de faire une première analyse visuelle de ces données.
Nous avons ensuite, à partir des hypothèses, cherché d'autres variables explicatives pouvant expliquer les variations d'activité sur le site.  

Dans un autre temps nous avons passé en revue quelques modèles de Machine learning afin de trouver le/les modèles les plus performants pour notre prédiction.
Nous avons dû developper des métriques spécifiques au projet afin de mesurer les modèles entre eux.
Pour finir nous présenterons une prédiction à partir de ces modèles.
'''

definition = '''
VOLVA  

Le mot « völva » viendrait de vǫlr, quenouille. Ce mot est à rapprocher du proto-germanique walwōn, qui donnera wand en anglais. La völva serait donc, comme les Nornes, une porteuse de quenouille.
Les völur, entre autres disciplines ésotériques traditionnelles, pratiquaient le seydr (enchantement), le spá (prophétie) et le galdr (magie runique, chamanisme).
Selon la mythologie et les récits historiques, les völur étaient censées posséder des pouvoirs tels qu'Odin lui-même, le père des dieux, faisait appel à leurs services pour connaître l'avenir des dieux :
 c'est notamment ce que rapporte la Völuspá, dont le titre lui-même, « völv-s-spá », se traduit par « chant de la prophétesse ».

'''

site= '''
le site logisitque est composé de plusieurs secteurs d'activités:  

- les Fruits et Legumes  
- le Frais  
- le Gel  
- le Sec  

  '''

#DATA


#DATA
datainit= '''
Data disponible au début du projet :  
  - Reporting à destination de la direction synthetisant les volumes et performances sous forme de fichier excel mensuel.  
Ce fichier regroupe par secteur :
  - le volume par jour
  - le nombre d'heures effectuées
  - la productivité
  '''

data1 = '''Les données d'origine du projet sont un ensemble de fichiers Excel qui regroupent les volumes et les performances par secteur.
Chaque fichier représente un mois. Grâce à un code Python les mois sont concaténés.
'''
data2 = '''
Les données sont ensuite nettoyées à travers :
 - une selection des variables intéressantes pour les analyses et prévisions
 - un traitement aucas par cas des erreurs
 - une analyse des outliers
 - une recherche des données manquantes
'''
data3 = '''
Réflexion des hypothèses impactant le volume et intégration des données permettant de valider ces hypothèses
 - position du jour dans l'espace temporel : Est-ce que la position du jour ou semaine dans le mois ou année impact le volume de commande?
 - jours fériés : Est-ce que la proximité d'un jour férié (passé ou à venir) impacte les commandes de supermarchés pour compenser le jour de fermeture ?
 - vacances scolaire par zone : 
   - Est-ce que le fait d'être en vacances impacte le volume de commande ?\n
   - Est-ce que la zone de vacances impacte le volume de commande?\n
   - Est-ce que la proximité (- de 7 jours) du début des vacances scolaires impacte le volume de commande.\n
 - températures moyennes saisonnières : Est-ce que les températures moyennes impactent le volume de commande?
 - les promotions : Est-ce que les promotions prévues impactent le volume de commande?
 - les semaines spéciales : Est-ce que les 2 premières semaines de juin et août ainsi que la dernière semaine de l'année impacte le volume de commande?
'''

data4 = '''
 Les données utiles en fonction des hypothèses sont nettoyées et ajoutées au DS de base.
'''

data5 = '''
 Le DF est prêt pour la modélisation et les prédictions.
'''

# Les données d'origine du projet sont un ensemble de fichiers Excel qui regroupent les volumes et les performances par secteur.
# Chaque fichier représente un mois.

# Les données ont été nettoyées et adaptées à l'exploitation pour analyse, visualisation, modelisation et prédiction.


# En fonction des hypothèses émises pour expliquer les variations de volume nous avons ajouté d'autres données externes à celles de départ:
# - position du jour dans l'espace temporel : Est-ce que la position du jour ou semaine dans le mois ou année impact le volume de commande?\n
# -> Nous avons donc intégré la position du jour dans la semaine, dans le mois et dans l'année ainsi que la position de la semaine dans le mois et dans l'année.


# - jours fériés : Est-ce que la proximité d'un jour férié (passé ou à venir) impacte les commandes de supermarchés pour compenser le jour de fermeture ? 
# -> Nous avons identifé si le jour férié le plus proche est passé ou à venir.


# - vacances scolaire par zone : /Est-ce que le fait d'être en vacances impacte le volume de commande ?\n
# /Est-ce que la zone de vacances impacte le volume de commande?\n
# /Est-ce que la proximité (- de 7 jours) du début des vacances scolaires impacte le volume de commande.\n
# -> Nous avons intégré le calendrier scolaire français par zone et identifié les jours proches (- de 7 jours) du début des vacaces scolaires


# - températures moyennes saisonnières : Est-ce que les températures moyennes impactent le volume de commande?
# -> Nous avons intégré les températures moyennes par jour depuis 2016 dans les régions concernées par le site logistique.


# - les promotions : Est-ce que les promotions prévues impactent le volume de commande?
# -> Nous avons intégré tous les types de promotions en cours.


# - les semaines spéciales : Est-ce que les 2 premières semaines de juin et août ainsi que la dernière semaine de l'année impacte le volume de commande?
# -> Nous avons remarqué que certaines semaines dans l'année ont un volume de commande important ainsi que la dernière semaine de l'année nous les avons donc identifié.
# '''







#visualisation

mobile = '''
Ci-dessous un graphique représentant les volumes de l'entrepôt par secteur ainsi que le volume total sur les années 2020 et 2021.  
Nous pouvons constater que les grandes variations du volumes sont cycliques.  
Nous retrouvons des périodes similaires d'une année sur l'autre.
Le secteur frais représente 80% du volume total donc impact fortement les variations du site.
'''

distrib = '''
Ci-dessous la distribution des volumes par secteur.
Nous pouvons constater que les secteurs GEL et FFL ont une distribution assez compacte et centré sur la moyenne/mediane ce qui nous amène à penser que le volume ne varie pas beaucoup
Par contre la distribution du frais est plus étalée ce qui induit un plus grand écart type et de fortes variations. 
'''

violo = '''
Représentation graphique des volumes journaliers avec distribution et écart-type.
les volumes journaliers nous montre que :
    -le secteur GEL est très homogene dans ses journées (sauf le samedi)
    -le secteur FFL a deux jours forts (jeudi et vendredi)
    -le secteur Frais à une journée très forte et homogène (le jeudi) mais les autres jours ont une distribution plus etalée (donc plus de variations)

'''


impact = '''
Les graphiques ci-dessous representent la variation des volumes la semaine precedant le jour ferié en fonction de la place du jour férié dans la semaine.
nous pouvons constater que le jour ferié impact fortement l'activité et plus particulierement sur certains jours en fonction du secteur et de la place du jour férié dans la semaine'
'''




#modelisation

dataselect = '''
Utilisation d'un modèle de Machine Learning (Régression) pour sélectionner les variables explicatives.

'''

modelregression = '''
modèles de regressions en test:  

GradientBoostRegressor  
Bayesian Ridge  
Lasso   
Random Forest Regressor  
KNeighbors Regressor  
ElasticNetCV  
Decision Tree Regressor  
'''

comparaison = '''
Comparaison des différents modèles de régression sur chaque jour de la semaine.  

Nous selectionnerons le meilleur modèle pour chaque jour.
'''


#prediction

periode = '''
Choisir :  
-le secteur d'activité  
-une date de début  
-une date de fin  
 afin d'afficher la prédiction et le besoin en opérateur logistique sur la période.
'''
