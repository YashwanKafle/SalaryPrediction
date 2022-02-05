

import streamlit as st
import pickle
import numpy as np 

def load_model():
    with open('salary_prediction_model.pkl', 'rb') as file:
        data = pickle.load(file)
    
    return data

data = load_model()


model = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]
le_devtype = data["le_devtype"]


st.title("Software Developer Salary Prediction")

dev_type = ['Data scientist or machine learning specialist',
    'Back-end Developer',
    'Desktop or Enterprise applications Developer',
    'Full-stack Developer', 'Other',
    'Embedded Applications or devices Developer', 'Mobile Developer',
    'Front-end Developer']

countries =['Sweden', 'Spain', 'Germany', 'Turkey', 'Canada', 'France',
    'Switzerland',
    'United Kingdom of Great Britain and Northern Ireland',
    'Russian Federation', 'Israel', 'other',
    'United States of America', 'Brazil', 'Italy', 'Netherlands',
    'Poland', 'Australia', 'India', 'Norway']

education = ['Master’s degree', 'Bachelor’s degree', 'Less than a Bachelors',
    'Post grad']
    


country = st.selectbox("Country",countries)

education = st.selectbox("Education label",education)

devtype = st.selectbox("Developement field",dev_type)

experience = st.slider("Years of Ecperience",0,50,3)


submit = st.button("Calculate salary")

if submit:
    X = np.array([[country, education, experience,devtype]])
    
    X[:, 0] = le_country.transform(X[:,0])
    X[:, 1] = le_education.transform(X[:,1])
    X[:,3] = le_devtype.transform(X[:,3])
    X = X.astype(float)

    salary = model.predict(X)

    st.write(f"The estimated salary is ${salary[0]:.2f}")

    