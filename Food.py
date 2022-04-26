import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.title("FOOD RECOMMENDER")

age=st.number_input("Age : ")
height=st.number_input("Height (in centimeters) :")
weight=st.number_input("Weight (in kilograms) :")

st.write("Requirement :")

data=pd.read_csv('food_Items.csv')
Fooditemsdata=data['Food_items']
Caloriedata=data['Calories']
Breakfastdata=data['Breakfast']
Lunchdata=data['Lunch']
Dinnerdata=data['Dinner']
Breakfastfood=[]
Lunchfood=[]
Dinnerfood=[]
BreakfastfoodID=[]
LunchfoodID=[]
DinnerfoodID=[]  
BreakfastdataNumpy=Breakfastdata.to_numpy()
DinnerdataNumpy=Dinnerdata.to_numpy()
LunchdataNumpy=Lunchdata.to_numpy()

def WeightLoss_Food():
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            Breakfastfood.append(Fooditemsdata[i])
            BreakfastfoodID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfood.append(Fooditemsdata[i])
            LunchfoodID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfood.append(Fooditemsdata[i])
            DinnerfoodID.append(i)
            
    LunchfoodIDdata = data.iloc[LunchfoodID]
    LunchfoodIDdata=LunchfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    LunchfoodIDdata=LunchfoodIDdata.iloc[l1append]
    LunchfoodIDdata=LunchfoodIDdata.T
 
    BreakfastfoodIDdata = data.iloc[BreakfastfoodID]
    BreakfastfoodIDdata=BreakfastfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    BreakfastfoodIDdata=BreakfastfoodIDdata.iloc[l1append]
    BreakfastfoodIDdata=BreakfastfoodIDdata.T
        
    DinnerfoodIDdata = data.iloc[DinnerfoodID]
    DinnerfoodIDdata=DinnerfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    DinnerfoodIDdata=DinnerfoodIDdata.iloc[l1append]
    DinnerfoodIDdata=DinnerfoodIDdata.T
    
    DinnerfoodIDdata=DinnerfoodIDdata.to_numpy() 
    LunchfoodIDdata=LunchfoodIDdata.to_numpy()
    BreakfastfoodIDdata=BreakfastfoodIDdata.to_numpy()
    
    bmi = weight/((height/100)**2) #calculating BMI
     
    for j in range (0,80,20):
        test_list=np.arange(j,j+20)
        for i in test_list: 
            if(i==age): 
                age2=round(j/20)    

    st.write("Body Mass Index :", bmi)
    if(bmi<16):
        st.write("BMI shows that you are Severely Underweight")
        bmi2=4
    elif(bmi>=16 and bmi<18.5):
        st.write("BMI shows that you are Underweight")
        bmi2=3
    elif(bmi>=18.5 and bmi<25):
        st.write("BMI shows that you are Healthy")
        bmi2=2
    elif(bmi>=25 and bmi<30):
        st.write("BMI shows that you are Overweight")
        bmi2=1
    elif(bmi>=30):
        st.write("BMI shows that you are Severely Overweight")
        bmi2=0
    
    avg=(bmi2+age2)/2
    CalorieData=DinnerfoodIDdata[1:,1:len(DinnerfoodIDdata)]  # K-Means Based  Dinner Food
    X = np.array(CalorieData)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    dlabel=kmeans.labels_               

    CalorieData=LunchfoodIDdata[1:,1:len(LunchfoodIDdata)] 
    X = np.array(CalorieData)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    llabel=kmeans.labels_ 
    
    CalorieData=BreakfastfoodIDdata[1:,1:len(BreakfastfoodIDdata)] 
    X = np.array(CalorieData)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    blabel=kmeans.labels_            
    
    datacal=pd.read_csv('calories.csv') 
    datacal1=datacal.T # train set
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightloss=datacal1.iloc[[1,2,7,8]]
    weightloss=weightloss.T
    weightlossNum=weightloss.to_numpy()
    weightloss=weightlossNum[1:,0:len(weightlossNum)]
    wlfinal=np.zeros((len(weightloss)*5,6),dtype=np.float32)
    
    t=0
    label=[]
    for i in range(5):
        for j in range(len(weightloss)):
            lis=list(weightloss[j])
            lis.append(bmicls[i])
            lis.append(agecls[i])
            wlfinal[t]=np.array(lis)
            label.append(blabel[j])
            t+=1
    
    X_test=np.zeros((len(weightloss),6),dtype=np.float32)
    #randomforest
    for j in range(len(weightloss)):
        lis=list(weightloss[j])
        lis.append(age2)
        lis.append(bmi2)
        X_test[j]=np.array(lis)*avg
    
    X_train=wlfinal # Features
    Y_train=label # Labels

    clf=RandomForestClassifier(n_estimators=100,random_state=None)
    clf.fit(X_train,Y_train)
    y_pred=clf.predict(X_test)

    st.write('WeightLoss FoodItem List')
    for i in range(len(y_pred)):
        if(y_pred[i]==2):
            if(i<88):
                st.write(Fooditemsdata[i],"-",Caloriedata[i]," calories")
                                           
def WeightGain_Food():
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            Breakfastfood.append(Fooditemsdata[i])
            BreakfastfoodID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfood.append(Fooditemsdata[i])
            LunchfoodID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfood.append(Fooditemsdata[i])
            DinnerfoodID.append(i)
            
    LunchfoodIDdata = data.iloc[LunchfoodID]
    LunchfoodIDdata=LunchfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    LunchfoodIDdata=LunchfoodIDdata.iloc[l1append]
    LunchfoodIDdata=LunchfoodIDdata.T
        
    BreakfastfoodIDdata = data.iloc[BreakfastfoodID]
    BreakfastfoodIDdata=BreakfastfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    BreakfastfoodIDdata=BreakfastfoodIDdata.iloc[l1append]
    BreakfastfoodIDdata=BreakfastfoodIDdata.T
        
    DinnerfoodIDdata = data.iloc[DinnerfoodID]
    DinnerfoodIDdata=DinnerfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    DinnerfoodIDdata=DinnerfoodIDdata.iloc[l1append]
    DinnerfoodIDdata=DinnerfoodIDdata.T
    
    DinnerfoodIDdata=DinnerfoodIDdata.to_numpy() 
    LunchfoodIDdata=LunchfoodIDdata.to_numpy()
    BreakfastfoodIDdata=BreakfastfoodIDdata.to_numpy()
        
    bmi = weight/((height/100)**2) 
           
    for j in range (0,80,20):
        test_list=np.arange(j,j+20)
        for i in test_list: 
            if(i == age): 
                age2=round(j/20)

    st.write("Body Mass Index : ",bmi)
    if(bmi<16):
        st.write("BMI shows that you are Severely Underweight")
        bmi2=4
    elif(bmi>=16 and bmi<18.5):
        st.write("BMI shows that you are Underweight")
        bmi2=3
    elif(bmi>=18.5 and bmi<25):
        st.write("BMI shows that you are Healthy")
        bmi2=2
    elif(bmi>=25 and bmi<30):
        st.write("BMI shows that you are Overweight")
        bmi2=1
    elif(bmi>=30):
        st.write("BMI shows that you are Severely Overweight")
        bmi2=0

    avg=(bmi2+age2)/2
    
    CalorieData=DinnerfoodIDdata[1:,1:len(DinnerfoodIDdata)] ## K-Means Based Food
    X = np.array(CalorieData)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    dlabel=kmeans.labels_
    
    CalorieData=LunchfoodIDdata[1:,1:len(LunchfoodIDdata)]
    X = np.array(CalorieData)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    llabel=kmeans.labels_
    
    CalorieData=BreakfastfoodIDdata[1:,1:len(BreakfastfoodIDdata)]
    X = np.array(CalorieData)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    blabel=kmeans.labels_
    
    datacal=pd.read_csv('calories.csv')
    datacal1=datacal.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightgain= datacal1.iloc[[0,1,2,3,4,7,9,10]]
    weightgain=weightgain.T
    weightgainNum=weightgain.to_numpy()
    weightgain=weightgainNum[1:,0:len(weightgainNum)]
    wgfinal=np.zeros((len(weightgain)*5,10),dtype=np.float32)
    
    r=0
    label=[]
    for i in range(5):
        for j in range(len(weightgain)):
            lis=list(weightgain[j])
            lis.append(bmicls[i])
            lis.append(agecls[i])
            wgfinal[r]=np.array(lis)
            label.append(llabel[j])
            r+=1

    X_test=np.zeros((len(weightgain),10),dtype=np.float32)

    for j in range(len(weightgain)):
        lis=list(weightgain[j])
        lis.append(age2)
        lis.append(bmi2)
        X_test[j]=np.array(lis)*avg
    
    X_train=wgfinal # Features
    Y_train=label # Labels
   
    clf=RandomForestClassifier(n_estimators=100,random_state=None)
    clf.fit(X_train,Y_train)    #Train the model using the training sets
    y_pred=clf.predict(X_test)
    
    st.write('WeightGain FoodItem List')
    for i in range(len(y_pred)):
        if y_pred[i]==1:
            if(i<88):
                st.write(Fooditemsdata[i],"-",Caloriedata[i]," calories")

def Healthy_Food():
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            Breakfastfood.append(Fooditemsdata[i])
            BreakfastfoodID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfood.append(Fooditemsdata[i])
            LunchfoodID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfood.append(Fooditemsdata[i])
            DinnerfoodID.append(i)
            
    LunchfoodIDdata = data.iloc[LunchfoodID]
    LunchfoodIDdata=LunchfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    LunchfoodIDdata=LunchfoodIDdata.iloc[l1append]
    LunchfoodIDdata=LunchfoodIDdata.T

    BreakfastfoodIDdata = data.iloc[BreakfastfoodID]
    BreakfastfoodIDdata=BreakfastfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    BreakfastfoodIDdata=BreakfastfoodIDdata.iloc[l1append]
    BreakfastfoodIDdata=BreakfastfoodIDdata.T
        
    DinnerfoodIDdata = data.iloc[DinnerfoodID]
    DinnerfoodIDdata=DinnerfoodIDdata.T
    l1=list(np.arange(5,15))
    l1append=[0]+l1
    DinnerfoodIDdata=DinnerfoodIDdata.iloc[l1append]
    DinnerfoodIDdata=DinnerfoodIDdata.T
    
    DinnerfoodIDdata=DinnerfoodIDdata.to_numpy() 
    LunchfoodIDdata=LunchfoodIDdata.to_numpy()
    BreakfastfoodIDdata=BreakfastfoodIDdata.to_numpy()
    
    bmi = weight/((height/100)**2) 
    for j in range(0,80,20):
        test_list=np.arange(j,j+20)
        for i in test_list: 
            if(i==age):
                age2=round(j/20)   

    st.write("Body Mass Index :", bmi)
    if(bmi<16):
        st.write("BMI shows that you are Severely Underweight")
        bmi2=4
    elif(bmi>=16 and bmi<18.5):
        st.write("BMI shows that you are Underweight")
        bmi2=3
    elif(bmi>=18.5 and bmi<25):
        st.write("BMI shows that you are Healthy")
        bmi2=2
    elif(bmi>=25 and bmi<30):
        st.write("BMI shows that you are Overweight")
        bmi2=1
    elif(bmi>=30):
        st.write("BMI shows that you are Severely Overweight")
        bmi2=0
        
    avg=(bmi2+age2)/2
    
    Datacalorie=DinnerfoodIDdata[1:,1:len(DinnerfoodIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3,random_state=0).fit(X)
    dlabel=kmeans.labels_
    
    Datacalorie=LunchfoodIDdata[1:,1:len(LunchfoodIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3,random_state=0).fit(X)
    llabel=kmeans.labels_
   
    Datacalorie=BreakfastfoodIDdata[1:,1:len(BreakfastfoodIDdata)]
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3,random_state=0).fit(X)
    blabel=kmeans.labels_
    
    datacal=pd.read_csv('calories.csv')
    datacal1=datacal.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    healthycat = datacal1.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    healthycatDdata=healthycat.to_numpy()
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]

    hfinal=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    s=0
    label=[]
    for i in range(5):
        for j in range(len(healthycat)):
            lis=list(healthycat[j])
            lis.append(bmicls[i])
            lis.append(agecls[i])
            hfinal[s]=np.array(lis)
            label.append(dlabel[j])
            s+=1
            
    X_test=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    for j in range(len(healthycat)):
        lis=list(healthycat[j])
        lis.append(age2)
        lis.append(bmi2)
        X_test[j]=np.array(lis)*avg
    
    X_train=hfinal # Features
    Y_train=label # Labels
    clf=RandomForestClassifier(n_estimators=100,random_state=None)
    clf.fit(X_train,Y_train)  #Train the model using the training sets
    y_pred=clf.predict(X_test)
    
    st.write("Healthy FoodItem List")
    for i in range(len(y_pred)):
        if y_pred[i]==2:
            if(i<88):
                st.write(Fooditemsdata[i],"-",Caloriedata[i]," calories")
                
res=st.button("Weight Loss")
res2=st.button("Healthy")
res1=st.button("Weight Gain")

if(res):
    WeightLoss_Food()
elif(res1):
    WeightGain_Food()
elif(res2):
    Healthy_Food()

