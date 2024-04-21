import streamlit as st
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from streamlit_lottie import st_lottie

input_size = 6


class LogisticRegression(torch.nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def decide(y):
    return 1. if y >= 0.5 else 0.


def load_model():
    with open('model1.pkl', 'rb') as file1:
        model1 = pickle.load(file1)
    
    with open('model2.pkl', 'rb') as file2:
        model2 = pickle.load(file2)

    return model1, model2

data1,data2=load_model()


def predict():
    col1, col2 = st.columns([0.3,0.7])
    with col1:
        url = requests.get("https://lottie.host/cfa6d512-d818-4b19-9cef-63e4fbfeffc4/yIe8LWxglL.json") 

        url_json = dict()

        if url.status_code == 200: 
            url_json = url.json() 
        else: 
            print("Error in the URL") 

        st_lottie(url_json,height=200, width=200, speed=1, loop=True) 

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.write("<h1 style='text-align: left;'>Health Predictor </h1>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: left; color:grey'>Predicting Health, Protecting Privacy </h5>", unsafe_allow_html=True)
        
    st.markdown("")
    st.markdown("")

    choice=("Select","Yes","No")

    temp=st.number_input("Temperature (in °C)")
    nau=1 if st.selectbox("Occurrence of Nausea",choice)=="Yes" else 0
    lp=1 if st.selectbox("Lumbar Pain",choice)=="Yes" else 0
    up=1 if st.selectbox("Urine Pushing",choice)=="Yes" else 0
    mp=1 if st.selectbox("Micturition Pains",choice)=="Yes" else 0
    bou=1 if st.selectbox("Burning of Urethra",choice)=="Yes" else 0

    st.markdown("")
    st.markdown("")

    st.markdown("""<style>div.stButton {text-align:center}</style>""", unsafe_allow_html=True)


    ok=st.button("Predict")
    st.markdown("")
    st.markdown("")
    if ok:
        x = [temp, nau, lp, up, mp, bou]
        ip = np.array([x],dtype = np.float32)
        validation = torch.tensor(ip[:, :6], dtype=torch.float32)
        predictionm1 = data1(validation)
        pr = predictionm1.tolist()
        pred1 = 'yes' if decide(predictionm1) else 'no'
        p1 = f'You might have Acute Inflammation in Urinary Bladder. The chances are {pr[0][0] * 100:.2f}%' if pred1 == 'yes' else "You don't have Acute Inflammation in Urinary Bladder"
        st.write("<h3 style='text-align: left; color:grey'>Acute Inflammation Predictor </h3>", unsafe_allow_html=True)
        st.subheader(f"{p1}")

        predictionm2 = data2(validation)
        pr2 = predictionm2.tolist()
        pred2 = 'yes' if decide(predictionm2) else 'no'
        p2 = f'You might have Nephritis in Renal Pelvis Origin. The chances are {pr2[0][0] * 100:.2f}%' if pred2 == 'yes' else "You don't have Nephritis in Renal Pelvis Origin"
        st.write("<h3 style='text-align: left;color:grey'>Nephritis Predictor </h3>", unsafe_allow_html=True)
        st.subheader(f"{p2}")


predict()
