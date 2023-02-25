#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd


# In[62]:


dfmcd = pd.read_csv(r"C:/Users/Amit/Downloads/mcdonalds.csv")


# In[63]:


dfmcd


# In[64]:


dfmcd.sample(5)


# In[65]:


dfmcd.isna().sum()


# In[66]:


dfmcd.info()


# In[67]:


dfmcd.shape


# In[68]:


dfmcd["Gender"].value_counts()


# In[69]:


dfmcd["VisitFrequency"].value_counts()


# In[70]:


dfmcd["Like"].value_counts()


# In[71]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


plt.pie(dfmcd['Gender'].value_counts(),labels=['Female','Male'],autopct='%0.1f%%')
plt.show()


# In[73]:


import seaborn as sns


# In[74]:


plt.figure(figsize=(25,8))
sns.countplot(x="Age",data=dfmcd,palette='hsv')
plt.grid()


# In[75]:


dfmcd['Like']= dfmcd['Like'].replace({'I hate it!-5': '-5','I love it!+5':'+5'})
 
sns.catplot(x="Like", y="Age",data=dfmcd,)
plt.title('Likelyness of McDonald w.r.t Age')
plt.show()


# In[76]:


from sklearn.preprocessing import LabelEncoder
def encoding(x):
    dfmcd[x] = LabelEncoder().fit_transform(dfmcd[x])
    return dfmcd

cat = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
       'tasty', 'expensive', 'healthy', 'disgusting']

for i in cat:
    encoding(i)
dfmcd


# In[77]:


dfmcd


# In[78]:


#Considering only first 11 attributes
df_eleven = dfmcd.loc[:,cat]
df_eleven


# In[79]:


#Considering only the 11 cols and converting it into array
x = dfmcd.loc[:,cat].values
x


# In[80]:


#Principal component analysis

from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_data = preprocessing.scale(x)

pca = PCA(n_components=11)
pc = pca.fit_transform(x)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
dfpc = pd.DataFrame(data = pc, columns = names)
dfpc


# In[81]:


#Proportion of Variance (from PC1 to PC11)
pca.explained_variance_ratio_


# In[82]:


np.cumsum(pca.explained_variance_ratio_)


# In[83]:


# correlation coefficient between original variables and the component

loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = df_eleven.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[84]:


#Correlation matrix plot for loadings 
plt.rcParams['figure.figsize'] = (20,15)
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()


# In[85]:


#Scree plot (Elbow test)- PCA
from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(10,5))


# In[87]:


# get PC scores
pca_scores = PCA().fit_transform(x)

# get 2D biplot
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=dfmcd.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(10,5))


# In[93]:


import warnings
warnings.filterwarnings('ignore')


# In[94]:


#Extracting segments

#Using k-means clustering analysis
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df_eleven)
visualizer.show()


# In[96]:


#K-means clustering 

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df_eleven)
dfmcd['cluster_num'] = kmeans.labels_ #adding to df
print (kmeans.labels_) #Label assigned for each data point
print (kmeans.inertia_) #gives within-cluster sum of squares. 
print(kmeans.n_iter_) #number of iterations that k-means algorithm runs to get a minimum within-cluster sum of squares
print(kmeans.cluster_centers_) #Location of the centroids on each cluster. 


# In[97]:


#To see each cluster size
from collections import Counter
Counter(kmeans.labels_)


# In[99]:


#Visulazing clusters
sns.scatterplot(data=dfpc, x="pc1", y="pc2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()


# In[101]:


#DESCRIBING SEGMENTS

from statsmodels.graphics.mosaicplot import mosaic
from itertools import product

crosstab =pd.crosstab(dfmcd['cluster_num'],dfmcd['Like'])
#Reordering cols
crosstab = crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4','+5']]
crosstab 


# In[102]:


#Mosaic plot gender vs segment
crosstab_gender =pd.crosstab(dfmcd['cluster_num'],dfmcd['Gender'])
crosstab_gender


# In[103]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[105]:


#box plot for age

sns.boxplot(x="cluster_num", y="Age", data=dfmcd)


# In[106]:


#Calculating the mean
#Visit frequency
dfmcd['VisitFrequency'] = LabelEncoder().fit_transform(dfmcd['VisitFrequency'])
visit = dfmcd.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
visit


# In[107]:


dfmcd['Like'] = LabelEncoder().fit_transform(dfmcd['Like'])
Like = dfmcd.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[108]:


#Gender
dfmcd['Gender'] = LabelEncoder().fit_transform(dfmcd['Gender'])
Gender = dfmcd.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
Gender


# In[109]:


segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit, on='cluster_num', how='left')
segment


# In[110]:


plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()


# In[ ]:




