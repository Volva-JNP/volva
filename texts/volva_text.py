# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 19:03:22 2022

@author: jk
"""

#home

intro = '''
# VOLVA PROJECT
Le projet a pour but d'analyser les historiques de volumes et de performance des différents 
services d’un site logistique pour en déduire des tendances et développer un outil prévisionnel afin d’adapter les ressources humaines nécessaires aux opérations logistiques avec une projection à plusieurs semaines.
Nous disposions des historiques de volumes de colis jour/jour sur la période 2020/2021, du nombre d’heures travaillées sur la même période et des notions de productivité par service (18 mois d’historique).
ces données sont issues d'une requete Oracle realisé mensuellement.
Nous avons donc dans un premier temps reuni l'ensemble des infos dans un seul dataset afin de faire une premiere analyse visuelle de ces données.
Nous avons ensuite, à partir des hypothèses,chercher d'autres variables explicatives pouvant expliquer les variations d'activité sur le site.
Dans un autre temps nous avons passer en revue quelques modeles de Machine learning afin de trouver le/les modeles les plus performants pour notre prediction.
nous avons dû devellopper des metriques specifiques au projet afin de mesurer les modeles entre eux.
Pour finir nous presenterons une prediction à partir de ses modeles.
'''




#DATA


DATA = '''

Les données d'origine du projet sont un ensemble de fichiers Excel qui regroupent les volumes et les performances par secteur.
Chaque fichier représente un mois.

'''







#visualisation

mobile = '''
Ci-dessous un graphique representant les volumes de l'entrepot par secteur ainsi que le volume total sur les années 2020 et 2021.
Nous pouvons constater que les grandes variations du volumes sont cyclique.Nous retrouvons des periodes similaire d'une année sur l'autre.
le secteur frais represente une forte proportion du volume totale donc impact fortement les variations du volume total
'''

distrib = '''
Ci_dessous la distribution des volumes par secteur.
Nous pouvons constater que les secteurs GEL et FFL ont une distribution assez compacte et centré sur la moyenne/mediane ce qui nous amene à penser que le volume ne varie pas beacoup
Par contre la distribution du frais est plus etalée ce qui induit un plus grand ecart type et de fortes variations.
'''

violo = '''
Représentation graphique des volumes journaliers avec distribution et ecart-type.
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




#prediction
