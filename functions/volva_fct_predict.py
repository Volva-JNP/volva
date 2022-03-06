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


def build_page_predict():

    with st.expander('Information'):    
        col1, col2 = st.columns(2)
        with col1:
            st.write("A compléter")
        with col2:
            st.write("A compléter")

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    menu_secteur  = st.radio(    
    "",
    ("vide", "secteur frais", "secteur GEL", "secteur FFL"),

    )        

    col_from, col_to = st.columns(2)
    with col_from:
        date_debut = st.date_input('Date de début')
    with col_to:
        date_fin = st.date_input('Date de fin')
    

    
