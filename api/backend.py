from fastapi import FastAPI, Body
import mfl as mfl
import pandas as pd
import numpy as np
import mfl.api.data_loaders as mfldata
import nfl_data_py as nfl
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, precision_recall_curve, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from xgboost import XGBClassifier, XGBRFClassifier

from catboost import CatBoostClassifier, CatBoostRegressor

import keras
from keras.layers import Dense, ReLU, Bidirectional, Normalization, Dropout, Input
from keras.models import Sequential

from bs4 import BeautifulSoup
import time

import seaborn as sns
import matplotlib.pyplot as plt

import math
import pickle

def map_years_with_draft_team(x):
        if x >= 4:
            return 1
        else: 
            return 0
        
def map_seasons_started(x):
        if x >= 3:
            return 1
        else: 
            return 0
    
def preprocess(df):
    df = df[df['draft_year'] <= 2019].dropna()
    numeric_features = ['G', 'Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'TD%', 'Int', 'Int%', 'Y/A', 'AY/A', 'Y/C', 'Y/G', 'Rate', 'college_seasons']
    categorical_features = ['recent_team']
    ordinal_features = ['round', 'pick']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('ord', OrdinalEncoder(), ordinal_features)
        ],
        remainder='drop' 
    )

    final_transformed = pd.DataFrame(preprocessor.fit_transform(df))
    final_transformed['seasons_with_draft_team'] = df['seasons_with_draft_team'].values

    return pd.DataFrame(final_transformed), preprocessor, numeric_features + categorical_features + ordinal_features
    
def score(y_test, y_probs, y_preds):
    accuracy = accuracy_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    roc_auc = roc_auc_score(y_test, y_probs)
    recall = recall_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)

    metric_dict = {
        'accuracy' : accuracy,
        'f1' : f1,
        'roc_auc': roc_auc,
        'recall': recall,
        'precision': precision
    }

    return metric_dict


def catboost(df, year_cutoff=2019, feature_set=None, kfold=False, folds=2):

    df = df.dropna()
    df = df[df['draft_year'] <= year_cutoff]

    if feature_set is None:
        X = df.drop(['player_name', 'pfr_player_name', 'seasons_with_draft_team', 'seasons_started'],axis=1)
        y = df['seasons_started']
    elif feature_set is not None:
        pass
    
    X = X
    y = y
    y_mapped = y.apply(map_years_with_draft_team)
    y_mapped = y.apply(map_seasons_started)

    X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=.25, stratify=y_mapped, shuffle=True)

    model = CatBoostClassifier(one_hot_max_size=15,
                                iterations=500, 
                                cat_features=X.select_dtypes(include='object').columns.tolist())
    
    model.fit(X_train, y_train)

    y_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:,1]

    metrics = score(y_test, y_probs, y_preds)
    
    model.fit(X, y_mapped)
    fit_model = model
    model_results = pd.DataFrame(metrics, index=[0])

    return model, model_results
    

def predict_2025_qb(model, player_name, round, pick, recent_team, predictors, season=2025):
    season = season
    variant_features = ['round', 'pick', 'draft_year']
    available_features = np.setdiff1d(model.feature_names_[:-1], variant_features).tolist()
    predictors = predictors[available_features]
    
    initial_features = pd.DataFrame({
        'round' : round,
        'pick' : pick,
        'draft_year' : season
    }, index=[0])
    
    processing = pd.concat([initial_features, predictors], axis=1)
    processing['recent_team'] = recent_team
    prediction = model.predict_proba(processing).tolist()[0][1]

    result_dict = {
         'player_name' : player_name,
         'round' : round,
         'pick' : pick,
         'prob' : prediction
    }
    
    return pd.DataFrame(result_dict, index=[0])

scaler_path = f'/Users/benstager/Desktop/mfl_project/mfl/api/saved_models/scaler1.pkl'
xgb_path = f'/Users/benstager/Desktop/mfl_project/mfl/api/saved_models/xgb1.pkl'
nn_path = f'/Users/benstager/Desktop/mfl_project/mfl/api/saved_models/nn.pkl'

with open(scaler_path, 'rb') as file_path:
    scaler = pickle.load(file_path)
with open(xgb_path, 'rb') as file_path:
    model = pickle.load(file_path)

app = FastAPI()

@app.post("/predict")
def predict(data: dict = Body(...)):

    player = data['player']
    team = data['team_feature']
    round = data['round']
    pick = data['pick']

    raw_data = pd.read_csv("~/Desktop/mfl_project/mfl/data/full_qb_dataset_v2.csv")
    select_feature = raw_data[raw_data['player_name'] == player]
    select_feature['recent_team'] = team
    print(team)
    select_feature['round'] = round
    print(round)
    select_feature['pick'] = pick
    features = pd.read_csv('/Users/benstager/Desktop/mfl_project/mfl/data/cols.csv').values[:,0]
    select_feature = select_feature[features]

    input_data_transformed = scaler.transform(select_feature)
    prediction = model.predict_proba(input_data_transformed).tolist()[0][1]
    print(prediction)

    return {"probability": prediction}