import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

df = pd.read_excel(r'E:\\Masai DA Projects\\Swiggy Project\\data_swiggy.xlsx')

# Data Preprocessing
df.drop_duplicates(inplace=True)
avg_df=pd.DataFrame(df.groupby(['cusines','location']).agg({'price_for_one':'mean'})).reset_index()

# Unique Locations
location_list=list(avg_df['location'].unique())
loc_list=sorted(location_list)

# Unique Cusines
cusine_list=list(avg_df['cusines'].unique())
cus_list=sorted(cusine_list)

# Encoding Categorical Data
loc_dit={}
for i in range(len(loc_list)):
    loc_dit[loc_list[i]]=i
cus_dit={}
for i in range(len(cus_list)):
    cus_dit[cus_list[i]]=i

# Taking each value from the columns
avg_df['location']=avg_df['location'].apply(lambda x:loc_dit[x])
avg_df['cusines']=avg_df['cusines'].apply(lambda x:cus_dit[x])
x=avg_df.drop(['price_for_one'],axis=1)
y=avg_df['price_for_one']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
sc=StandardScaler()
x_sc=sc.fit_transform(x_train)
x_test_sc=sc.transform(x_test)
new_df=avg_df.copy()

model=LinearRegression()
model.fit(x_sc,y_train)
y_pred=model.predict(x_test_sc)
print(metrics.r2_score(y_test,y_pred))
print(metrics.r2_score(y_test,y_pred))
pickle.dump(model,open('model_pred.pkl','wb'))

