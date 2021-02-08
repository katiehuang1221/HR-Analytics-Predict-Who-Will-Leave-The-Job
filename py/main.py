import streamlit as st
import pandas as pd

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

with header:
    st.title('Welcome to Smart HR!')

with dataset:
    st.header('HR Analytics Dataset')

    df_train = pd.read_csv('../dump/')

with features:
    st.header('Info of Candidates')

with model_training:
    st.header('Model')
    st.text('Choose the feature(s) you want to use for filtering!')