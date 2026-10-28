import streamlit as st
import pandas as pd
import tensorflow
import pickle as pkl
import numpy as np
import random

model = tensorflow.keras.models.load_model("credit_card_fraud_detection.keras")
scaler = pkl.load(open("scaler (1).pkl", "rb"))

st.title("banknote_real_or_fake")
vWTI = st.number_input("VWTI")
sWTI = st.number_input("SWTI")
cWTI = st.number_input("CWTI")
eI = st.number_input("EI")

if st.button("Predict"):
    df = pd.DataFrame([{
        "VWTI": vWTI,
        "SWTI": sWTI,
        "CWTI": cWTI,
        "EI": eI
    }])
    df = scaler.transform(df)
    res = model.predict(df)
    result = np.where(res>0.5, 1, 0)
    if result == 0 :
        st.success("Banknote Real")
    else :
        st.warning("Banknote Fake")
    if result == 0 :
        ran = random.choice([0,1,2,3,4])

        for i in range (0,4):

            if ran == 1 :
                st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRUeNfQtsf2SEbpPUw0VTDL8xbU4F9oZEF6Q4KCIcqhrw&s=10","real")
                break
            elif ran == 2 :
                st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQyJ3WK8vygZEcrVDP-XsVlsFqPZydZB660Q3vEnqMwDYOZUWW9S5FE0BVv&s=10","100")
                break
            else :
                st.image("https://c8.alamy.com/comp/2XK841E/appearance-comparison-between-old-and-new-one-yen-front-view-flat-lay-1-japanese-yen-note-and-1-japanese-yen-coin-in-white-background-2XK841E.jpg","real")
                break
    else :
        ran = random.choice([0,1,2])
        if ran == 0 :
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRP_Ts6prVJCQ5ZOaoIZ6olAOO6VZ5x4CEwbBIgSgbBAe2uJtNC0Nxyjjo&s=10","fake")
        elif ran == 1:
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQXEOcFISQNsFOm1ovsB3NLDRE2XJAiIVr1NEpD7FSjor10UN_lcQ2LiX0&s=10","fake")
        else :
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWnN8PTb20NtnvsLH_11MHaaTgZnRJu3S0zjiVQzG8zvx8NDij5msnzik&s=10","fake")


