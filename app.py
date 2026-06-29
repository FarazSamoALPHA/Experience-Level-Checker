import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

d=pickle.load(open('Ex_model_dict.pkl','rb'))
df= pd.DataFrame(d)

model=pickle.load(open('model.pkl','rb'))

col_tranfr = pickle.load(open('transform.pkl','rb'))

rf = RandomForestRegressor()

def predicition(Age, Gender, Weight_kg, Height_m, Max_BPM, Avg_BPM, Resting_BPM, Session_Duration_hours, Calories_Burned, Workout_Type, Fat_Percentage, Water_Intake_liters, Workout_Frequency_days_week, BMI):
  features=np.array([[Age, Gender, Weight_kg, Height_m, Max_BPM, Avg_BPM, Resting_BPM, Session_Duration_hours, Calories_Burned, Workout_Type, Fat_Percentage, Water_Intake_liters, Workout_Frequency_days_week, BMI]])
  transform_fe= col_tranfr.transform(features)
  pred_valu = rf.predict(transform_fe)
  return pred_valu[0]


st.title('Experience Level')


option=st.selectbox(
'Enter Value of Age',df['Age'].values)

option1=st.selectbox(
'Enter Value of Gender',df['Gender'].values)

option2=st.selectbox(
'Enter Value of Weight_kg',df["Weight (kg)"].values)

option3=st.selectbox(
'Enter Value of Height_m',df['Height (m)'].values)

option4=st.selectbox(
'Enter Value of Max_BPM',df['Max_BPM'].values)

option5=st.selectbox(
'Enter Value of Avg_BPM',df['Avg_BPM'].values)

option6=st.selectbox(
'Enter Value of Resting_BPM',df['Resting_BPM'].values)

option7=st.selectbox(
'Enter Value of Session_Duration_hours',df['Session_Duration (hours)'].values)

option8=st.selectbox(
'Enter Value of Calories_Burned',df['Calories_Burned'].values)

option9=st.selectbox(
'Enter Value of Workout_Type',df['Workout_Type'].values)

option10=st.selectbox(
'Enter Value of Fat_Percentage',df['Fat_Percentage'].values)

option11=st.selectbox(
'Enter Value of Water_Intake_liters',df['Water_Intake (liters)'].values)

option12=st.selectbox(
'Enter Value of Workout_Frequency_days_week',df['Workout_Frequency (days/week)'].values)

option13=st.selectbox(
'Enter Value of BMI',df['BMI'].values)

st.button("Reset", type="primary")
if st.button("Check Experience"):
    query = np.array([[option,option1,option2,option3,option4,option5,option6,option7,option8,option9,option10,option11,option12,option13]])
    transform_da = col_tranfr.transform(query)
    st.title('Your Experience Level :'+str(model.predict(transform_da)[0]))
else:
    st.write("Goodbye")

