import pandas as pd
import numpy as np
data=pd.read_csv("OnlineRetail.csv",encoding='Iso-8859-1')
print(data)
print(data.head)
print(data.shape)
print(data.info())
print(data.columns)
print(data.describe)
print(data.isnull().sum())
data_null=round(100*(data.isnull().sum())/len(data),2)
print(data_null)
#Dropping rows having missing values
data.drop(["StockCode"], axis=1,inplace=True)
print(data.columns)
data["CustomerID"]=data["CustomerID"].astype(str)
print(data)
#new attribute
data["Amount"]=data["Quantity"]*data["UnitPrice"]
rfm_m=data.groupby("CustomerID")["Amount"].sum()
rfm_m=rfm_m.reset_index()
print(rfm_m.head())
#by monetary
rfm_m=rfm_m.reset_index()
print(rfm_m)
# Grouping by Country and calculating total sales
sales_by_country = data.groupby('Country')['Quantity'].sum().sort_values(ascending=False)
print('The countries selling the most are:')
print(sales_by_country.head())

# Grouping by Description and calculating total quantity sold
sales_by_product = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print('The products selling the most are:')
print(sales_by_product.head())
#frequently sold product
Frequently_by_product = data.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False)
print('The frequent sold product:')
print(sales_by_product.head())
# sales for the last month 
daily_sales_previous_month = data.groupby('InvoiceDate')['Amount'].sum().sort_values(ascending=False)
print(daily_sales_previous_month)
#convert the invoicedate column
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'],format='%m/%d/%Y %H:%M')
print(data['InvoiceDate'])
#find maximum date
max_date=max(data["InvoiceDate"])
print(max_date)
#find minimum date
min_date=min(data["InvoiceDate"])
print(min_date)
# Find the difference between max date and min date 
data["diff"]=max_date-min_date
print(data["diff"])
#Transaction for the last 30 days
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'],format ='%d-%m-%Y %H:%M')
last_date = data['InvoiceDate'].max()
month_start_day= last_date - pd.Timedelta(days=30)
print(month_start_day)

# Filter data for the last 30 days
last_month_data = data[data['InvoiceDate'] >= month_start_day]

# Calculate the total sales amount for the last 30 days
total_sales_last_month = last_month_data['Quantity'].sum()
print(total_sales_last_month)
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.plotting import scatter_matrix

data.plot(kind='box',subplots=True,sharex=False,sharey=False)
plt.show()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(data[['Quantity','UnitPrice']],data[['Description']],test_size=0.3,random_state=0)
from sklearn import preprocessing
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)
from sklearn.cluster import KMeans
data_3=KMeans(n_clusters = 3,random_state =0,n_init='auto')
data_3.fit(x_train_norm)
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=data_3.labels_)
plt.show()
from sklearn.metrics import silhouette_score
perf=(silhouette_score(x_train_norm,data_3.labels_,metric='euclidean'))
print(perf)
'''Testing a number of clusters to determine how many to use'''
K=range(2,8)
fit=[]
score=[]
for k in K:
    '''Train the model for the current value of k on the training model'''
    model=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(x_train_norm)
    fit.append(model)
    score.append(silhouette_score(x_train_norm,model.labels_,metric='euclidean'))
print(fit)
print(score)
'''Visualize the models for k=2,k=4,k=7,k=5'''
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=fit[0].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=fit[2].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=fit[5].labels_)
plt.show()
sns.scatterplot(data=x_train,x='Quantity',y='UnitPrice',hue=fit[0].labels_)
plt.show()
sns.lineplot(x=K,y=score)
plt.show()
sns.scatterplot(data=x_train,x='UnitPrice',y='Quantity',hue=fit[3].labels_)
plt.show()
sns.boxplot(x=fit[3].labels_, y=y_train['Description'])
plt.show()



    















                                   






















