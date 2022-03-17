import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import squarify
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import streamlit as st
import plotly.express as px
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Read data
# df = pd.read_csv('OnlineRetail.csv', header= 0, encoding= 'unicode_escape')

#--------------
# GUI
st.title("Data Science Project")
st.write("## Customer Segmentation")

# Upload file
# Upload file/ Read file
uploaded_file = st.file_uploader('Choose a file', type = ['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='unicode_escape')
    df.to_pickle("OnlineRetail_new.gzip", compression='gzip')
else:
    df = pd.read_pickle("OnlineRetail.gzip",compression='gzip')

# Removing duplicate entries.
print('Duplicate entries: {}'.format(df.duplicated().sum()))
print('{}% rows are duplicate.'.format(round((df.duplicated().sum()/df.shape[0])*100),2))
df.drop_duplicates(inplace = True)

# Checking country wise distribution of transactions.
temp = df.groupby(['Country'],as_index=False).agg({'InvoiceNo':'nunique'}).rename(columns = {'InvoiceNo':'Orders'})
total = temp['Orders'].sum(axis=0)
temp['%Orders'] = round((temp['Orders']/total)*100,4)
temp.sort_values(by=['%Orders'],ascending=False,inplace=True)
temp.reset_index(drop=True,inplace=True)

# Removing cancelled orders from the data.
invoices = df['InvoiceNo']
x = invoices.str.contains('C', regex=True)
x.fillna(0, inplace=True)
x = x.astype(int)
x.value_counts()
df['order_canceled'] = x
df.head()

df['order_canceled'].value_counts()

# We find out that CustomerID values are missing for those customers which have negative quantity values. Therefore, we will remove them too.
df = df[df['CustomerID'].notna()]
df.reset_index(drop=True,inplace=True)
df_uk = df[df.Country == 'United Kingdom']
data = df_uk[['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']]

all_dates = (pd.to_datetime(data['InvoiceDate'])).apply(lambda x:x.date())
(all_dates.max() - all_dates.min()).days

data['InvoiceMonth'] = data['InvoiceDate'].apply(lambda x: pd.Timestamp(x).strftime('%d-%m-%Y'))
grouping = data.groupby('CustomerID')['InvoiceMonth']
data['InvoiceMonth'] = grouping.transform('min')

# RFM Segmentation
data['TotalSum'] = data['Quantity'] * data['UnitPrice']
data[data['TotalSum']> 17.850000].sort_values('TotalSum',ascending=False)
data['InvoiceDate'] = data['InvoiceDate'].astype('datetime64[ns]')

# RFC
# Convert string to date, get max date of dataframe
max_date = data['InvoiceDate'].max().date()

Recency = lambda x : (max_date - x.max().date()).days
Frequency  = lambda x: len(x.unique())
Monetary = lambda x : round(sum(x), 2)

df_RFM = data.groupby('CustomerID').agg({'InvoiceDate': Recency,
                                        'InvoiceNo': Frequency,  
                                        'TotalSum': Monetary })

# Rename the columns of Dataframe
df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
# Descending Sorting
df_RFM = df_RFM.sort_values('Monetary', ascending=False)

# Create labels for Recency, Frequency, Monetary
r_labels = range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
f_labels = range(1, 5)
m_labels = range(1, 5)

# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=r_labels)

f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=f_labels)
 
m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=m_labels)

# Create new columns R, F, M
df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)

def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)

rfm_count_unique = df_RFM.groupby('RFM_Segment')['RFM_Segment'].nunique()

# Calculate RFM_Score
df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)

df_RFM.groupby('RFM_Score').agg({'Recency': 'mean',
                                   'Frequency': 'mean',
                                   'Monetary': ['mean', 'count'] }).round(1)

def rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Top'
    elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 9):
        return 'Middle'
    else:
        return 'Low'

# Create a new column RFM_Level
df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)

# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_agg = df_RFM.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

# Reset the index
rfm_agg = rfm_agg.reset_index()

df_now = df_RFM[['Recency', 'Frequency','Monetary']]

# Build model with k=3
model_1 = KMeans(n_clusters=3, random_state=42)
model_1.fit(df_now)

# Build model with k = 4
model_2 = KMeans(n_clusters=4, random_state=42)
model_2.fit(df_now)

#5. Save models
# luu model classication
pkl_filename = "KMeans_3.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(model_1, file)
  
# luu model CountVectorizer (count)
pkl_filename_1 = "KMeans_4.pkl"  
with open(pkl_filename_1, 'wb') as file:  
    pickle.dump(model_2, file)


#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    KMeans_3_model = pickle.load(file)
# doc model count len
with open(pkl_filename_1, 'rb') as file:  
    KMeans_4_model = pickle.load(file)


# GUI
menu = ["Business Objective","RFM Segmentation","RFM KMeans","New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':
    st.subheader("Business Objective")
    st.write("""###### => RFM segmentation is a convenient method for segmenting customers. The output is intuitive, making it easy for marketing to understand and interpret later. The calculation of RFM segmentation pays attention to three factors (Recency, Frequency, and Monetary Value).""")
    st.image("Unsupervised Segments.png")
    st.write("""###### => Problem/ Requirement: Use RFM to choose Segment Customers.""")

elif choice == 'RFM Segmentation':
    st.subheader("RFM Segmentation")
    st.write("#### 1. Some data")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))
    
    st.write("#### 2. Visualization")
    plt.figure(figsize=(13,6))
    fig1 = sns.barplot(x="Country",y="%Orders",data=temp[:10])
    for p in fig1.patches:
        fig1.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    plt.xlabel("Country", size=14)
    plt.ylabel("%Orders", size=14)
    st.pyplot(fig1.figure)

    st.write("#### 3. Checking the total number of products, transactions and customers.")
    st.dataframe(pd.DataFrame([{'products': len(df['StockCode'].value_counts()),
                        'transactions': len(df['InvoiceNo'].value_counts()),
                        'customers': len(df['CustomerID'].value_counts()),}],
                        columns = ['products', 'transactions', 'customers'], index = ['quantity']))
    st.write("#### 4. Evaluation")
    n1 = df['order_canceled'].value_counts()[1]
    n2 = df.shape[0]
    st.code('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))
    df = df.loc[df['order_canceled'] == 0,:]
    df.reset_index(drop=True,inplace=True)
    
    st.write("##### Describe:")
    st.code(df.describe())
    
    st.write("##### df_uk")
    st.dataframe(df_uk.head())
    
    st.write("##### Checking for nulls in the data.")
    st.code(data.isnull().sum())
    
    st.write("##### Start and end dates")
    st.code('Start date: {}'.format(all_dates.min()))
    st.code('End date: {}'.format(all_dates.max()))

    st.write("##### Let’s take a closer look at the data we will need to manipulate")
    st.code('Transactions timeframe from {} to {}'.format(data['InvoiceDate'].min(), data['InvoiceDate'].max()))
    st.code('{:,} transactions don\'t have a customer id'.format(data[data.CustomerID.isnull()].shape[0]))
    st.code('{:,} unique customer_id'.format(len(data.CustomerID.unique())))

    st.write("##### df_RFM")
    st.dataframe(df_RFM.head())
    
    st.write("##### df_RFM shape")
    st.code(df_RFM.shape)

    st.write("##### Virsulization")
    fig = plt.figure(figsize=(8,10))
    plt.subplot(3, 1, 1)
    sns.distplot(df_RFM['Recency'])# Plot distribution of R
    plt.subplot(3, 1, 2)
    sns.distplot(df_RFM['Frequency'])# Plot distribution of F
    plt.subplot(3, 1, 3)
    sns.distplot(df_RFM['Monetary']) # Plot distribution of M
    st.pyplot(fig)

    st.write("##### Count value")
    st.code(df_RFM['RFM_Level'].value_counts())

    st.write("##### rfm_agg")
    st.code(rfm_agg)

    st.write("##### Customers Segments")
    #Create our plot and resize it.
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)

    colors_dict = {'Low':'yellow','Middle':'royalblue', 'Top':'cyan'}

    squarify.plot(sizes=rfm_agg['Count'],
              text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
              color=colors_dict.values(),
              label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                      for i in range(0, len(rfm_agg))], alpha=0.5 )
    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    st.pyplot(fig)

    fig = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Level",
           hover_name="RFM_Level", size_max=100)
    st.plotly_chart(fig)

    fig = px.scatter_3d(df_RFM, x='Recency', y='Frequency', z='Monetary',
                    color = 'RFM_Level', opacity=0.5,
                    color_discrete_map = colors_dict)
    fig.update_traces(marker=dict(size=5),selector=dict(mode='markers'))
    st.plotly_chart(fig)

elif choice == 'RFM KMeans':
    df_now = df_RFM[['Recency', 'Frequency','Monetary']]
    sse = {}
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_now)
        sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

    fig = plt.figure(figsize=(8,6))
    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    st.pyplot(fig.figure)   

    

    df_now["Cluster"] = model_1.labels_
    summary_k3 = df_now.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':['mean', 'count']}).round(2)
    
    st.write("##### Build model with k = 3")
    st.code(summary_k3)

    # Calculate average values for each RFM_Level, and return a size of each segment 
    rfm_agg_k3 = summary_k3
    rfm_agg_k3.columns = rfm_agg_k3.columns.droplevel()
    rfm_agg_k3.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg_k3['Percent'] = round((rfm_agg_k3['Count']/rfm_agg_k3.Count.sum())*100, 2)

    # Reset the index
    rfm_agg_k3 = rfm_agg_k3.reset_index()

    # Change thr Cluster Columns Datatype into discrete values
    rfm_agg_k3['Cluster'] = 'Cluster '+ rfm_agg_k3['Cluster'].astype('str')

    # Print the aggregated dataset
    st.code(rfm_agg_k3)

    #Create our plot and resize it.
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)

    colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan'}

    squarify.plot(sizes=rfm_agg_k3['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict2.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg_k3.iloc[i])
                        for i in range(0, len(rfm_agg_k3))], alpha=0.5 )


    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    st.pyplot(fig)

    fig = px.scatter(rfm_agg_k3, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
           hover_name="Cluster", size_max=100)
    st.plotly_chart(fig)

    fig = px.scatter_3d(rfm_agg_k3, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
                    color = 'Cluster', opacity=0.3)
    fig.update_traces(marker=dict(size=20),selector=dict(mode='markers'))
    st.plotly_chart(fig)

    df_now["Cluster"] = model_2.labels_
    summary_k4 =df_now.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'Monetary':['mean', 'count']}).round(2)
    
    st.write("##### Build model with k = 4")
    st.code(summary_k4)

    # Calculate average values for each RFM_Level, and return a size of each segment 
    rfm_agg_k4 = summary_k4
    rfm_agg_k4.columns = rfm_agg_k4.columns.droplevel()
    rfm_agg_k4.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg_k4['Percent'] = round((rfm_agg_k4['Count']/rfm_agg_k4.Count.sum())*100, 2)

    # Reset the index
    rfm_agg_k4 = rfm_agg_k4.reset_index()

    # Change thr Cluster Columns Datatype into discrete values
    rfm_agg_k4['Cluster'] = 'Cluster '+ rfm_agg_k4['Cluster'].astype('str')

    # Print the aggregated dataset
    st.code(rfm_agg_k4)

    #Create our plot and resize it.
    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)

    colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                'Cluster3':'red'}

    squarify.plot(sizes=rfm_agg_k4['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict2.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg_k4.iloc[i])
                        for i in range(0, len(rfm_agg_k4))], alpha=0.5 )


    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    st.pyplot(fig)

    fig = px.scatter(rfm_agg_k4, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
           hover_name="Cluster", size_max=100)
    st.plotly_chart(fig)

    fig = px.scatter_3d(rfm_agg_k4, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
                    color = 'Cluster', opacity=0.3)
    fig.update_traces(marker=dict(size=20),
                  
                  selector=dict(mode='markers'))
    st.plotly_chart(fig)


elif choice == 'New Prediction':
    # Build model with k=5
    k = st.slider("Chọn k", 2, 10, 5, 1)
    # Model
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_now)
    df_now["Cluster"] = model.labels_
    # Calculate average values for each RFM_Level, and return a size of each segment 
    rfm_agg2 = df_now.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

    # Reset the index
    rfm_agg2 = rfm_agg2.reset_index()
    # Change thr Cluster Columns Datatype into discrete values
    rfm_agg2['Cluster'] = 'Cluster '+ rfm_agg2['Cluster'].astype('str')
    # Kết quả phân nhóm
    st.write("##### RFM-Kmeans Results:")
    st.dataframe(rfm_agg2)
    #Tree map.
    st.write("##### Tree map (RFM-Kmeans)")
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(14, 10)    
    colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

    squarify.plot(sizes=rfm_agg2['Count'],
                text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                color=colors_dict2.values(),
                label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                        for i in range(0, len(rfm_agg2))], alpha=0.5 )


    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    plt.axis('off')
    st.pyplot(fig)

    # Scatter Plot
    st.write("##### Scatter Plot (RFM-Kmeans)")
    # plt.clf()
    
    fig = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
        hover_name="Cluster", size_max=100)
    st.plotly_chart(fig)

    # 3D Scatter Plot
    st.write("##### 3D Scatter Plot (RFM)")
    fig = px.scatter_3d(rfm_agg2, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
                    color = 'Cluster', opacity=0.3)
    fig.update_traces(marker=dict(size=20),
                
                selector=dict(mode='markers'))
    st.plotly_chart(fig)
