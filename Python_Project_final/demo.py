#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import streamlit as st
import pickle
import numpy as np

st.markdown("<h1 style='text-align: center; color: black;'>Demo</h1>",unsafe_allow_html=True)
st.title("Data Visualization")
text = ["MaxDelq2PublicRecLast12M","MaxDelqEver"]

choice = {'derogatory comment':0,'120+ days delinqent':1,'90 days delinquent':2,
          '60 days delinquent':3,'30 days delinquent':4,'unknown delinquency':5,'current and never delinquent':6
          ,'all other':7
}

options = {
    "Condition not Met (e.g. No Inquiries, No Delinquencies) ": -7,
    "No Usable/Valid Trades or Inquiries": -8,
    "No Bureau Record or No Investigation": -9,
}
df_row = pd.read_csv("df_cleaned.csv")
df = df_row.iloc[:,:23]
df["RiskPerformance"] = df_row["RiskPerformance"]
columns=df.columns[:23].values


st.sidebar.write("Enter the number of rows to view")

rows = st.sidebar.number_input("", min_value=0,value=5)
cols = st.sidebar.multiselect('', columns)


df1=df.copy()


if cols:
    for k in range(len(cols)):
        st.sidebar.write("Input range of value for ",cols[k])
        if cols[k] not in text:           
           
            minimal = df1[cols[k]].min()
            maximal =df1[cols[k]].max()
            status = st.sidebar.slider('',float(minimal),float(maximal),(float(minimal),float(minimal)),key=k+1)
            st.sidebar.write(status)
            line= df1[cols[k]].apply(lambda x: status[0]< int(x) < status[1])
            df1 = df1.loc[line]
        else: 
            line = st.sidebar.selectbox("", list(choice.items()),format_func = lambda item: item[0],key = k+41)[1]
            df1 = df1.loc[df1[cols[k]]==line]

if rows > 0:
    st.dataframe(df1.head(rows))
    

st.write("Percentage of good in chosen criteria is", (1-((df1["RiskPerformance"]).mean())))


##data input

st.title("Input data below to evaluate the risk")

values= []


customer= st.text_input("Customer ID/Name")

for i in columns:
    st.header("%s"%(i))
    missing_value = 0
    if(i not in text):
        x=st.number_input('',value=int(df[i].mean()),key="2"+i)
        st.write("Check if data is missing")
        missing =st.checkbox("",key = i)
        if missing:
            
            st.write("Choose best description below")
            x = st.selectbox("", list(options.items()),format_func = lambda item: item[0],key = '40'+i)[1]
    else: 
        st.write("Choose best description below")
        x = st.selectbox("", list(choice.items()),format_func = lambda item: item[0],key = '40'+i)[1]
        
    values.append(x)





###Predict the value/Need to modified the input based on the model
input_df= pd.DataFrame(values,index= columns).T
st.dataframe(input_df)


## Data transformation same as preprocessing
df_row = pd.read_csv("heloc_dataset_v1.csv").iloc[:,1:]
filename = 'preprocessing.sav'
with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)

pipe=loaded_model[0]
pipe.fit(df_row)
df = pipe.transform(input_df)
output_df=pd.DataFrame(df,columns = loaded_model[1])

# Get dummy variables fron categorical data

ls1=[]
for i in range(8):
        ls1.append("%s = %d" %("MaxDelq2PublicRecLast12M",i)) 
MaxDelq2PublicRecLast12M_table = pd.DataFrame(np.zeros(8).astype(int),index=ls1).T
MaxDelq2PublicRecLast12M_table.iloc[:,output_df["MaxDelq2PublicRecLast12M"].astype(int)]=1

ls2=[]
for i in range(7):
        ls2.append("%s = %d" %('MaxDelqEver',i))

MaxDelqEver_table= pd.DataFrame(np.zeros(7).astype(int),index=ls2).T
MaxDelqEver_table.iloc[:,output_df['MaxDelqEver'].astype(int)]=1

final_df = pd.concat([output_df,MaxDelq2PublicRecLast12M_table,MaxDelqEver_table],axis=1)



filename = 'finalized_model.sav'
with open(filename, 'rb') as f:
	loaded_model = pickle.load(f)
    
yp = loaded_model.predict(final_df)


comfirm = st.button("Click to show prediction")

if yp:
    output = "Bad"
else:
    output="Good"
if comfirm:
    st.write("The predict value of the model: ",output, yp)
    


## out put to csv file

save = st.button("Click to save data into file")
if save:
   input_df["Predict"] = yp
   input_df.index = [customer]
   input_df.to_csv('customer.csv', mode='a', header=False)
