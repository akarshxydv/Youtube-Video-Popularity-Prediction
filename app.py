import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
st.header("Youtube popularity prediction")

# Load the model
le = pickle.load(open("labelEncoder.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
clf = pickle.load(open("clf.pkl", "rb"))


# Input fields
title = st.text_input("Title")
tags = st.text_input("Tags")
viewCount = int(st.number_input("View Count"))
commentCount = int(st.number_input("Comment Count"))

def features():
    feature = pd.DataFrame({
        "title": [title],
        "tags": [tags],
        "viewCount": [viewCount],
        "commentCount": [commentCount],
    })
    return feature


# Predict function
def predict():
    df = features()
    df['title'] = le.fit_transform(df['title'])
    df['tags'] = le.fit_transform(df['tags'])
    prediction = model.predict(df)
    return prediction

def predictAda():
    df = features()
    df['title'] = le.fit_transform(df['title'])
    df['tags'] = le.fit_transform(df['tags'])
    prediction = clf.predict(df)
    return prediction

if st.button("Predict"):
    result = predict()
    df = features()
    if result == 0:
        st.write("Not popular")
    else:
        st.write("Popular")
if st.button("Predict with ADA Boost"):
    result = predictAda()
    df = features()
    if result == 0:
        st.write("Not popular")
    else:
        st.write("Popular")


if st.button("Visuals"):
    st.image('output.png', caption=' caption')