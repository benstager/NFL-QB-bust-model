import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd 
import requests

def post_to_backend(data):
    url = "http://localhost:8000/predict"  # Replace with your backend endpoint
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=data, headers=headers)
    return response.json()

current_dir = os.path.dirname('app.py')
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

raw_data = pd.read_csv("~/Desktop/mfl_project/mfl/data/full_qb_dataset_v2.csv")
teams = raw_data['recent_team'].dropna().unique()
players = raw_data['player_name'].unique()

st.title('NFL Draft What If?')

st.header('QB Crystal Ball')

player = st.selectbox("Select a player:", players)
team_feature = st.selectbox("Change draft team:", teams)
round = st.selectbox("Change draft round:", range(1, 8))
pick = st.selectbox("Change draft pick:", range(1, 33))

input_data = {
    "player": player,
    "team_feature": team_feature,
    "round": round,
    "pick": pick
}

# scaled_data = scaler.transform(np.array([[feature1, feature2, feature3, feature4,feature5]]).reshape(1,-1))

if st.button("Predict"):
    response = post_to_backend(data=input_data)
    output_prob = response['probability']
    success_message = f"{player} drafted by {team_feature} in round {round} at pick {pick} has a {100 - (100*output_prob):.2f}% chance of being a bust"
    st.success(success_message)


