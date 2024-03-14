#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[45]:


df = pd.read_csv("covid19_Confirmed_dataset.csv")


# In[46]:


df.head()


# In[47]:


df.shape


# In[48]:


df.drop(["Lat","Long"],axis=1,inplace=True)


# In[52]:


df.drop(['1/22/20','1/23/20','1/24/20','1/25/20','1/26/20'],axis=1,inplace=True)


# In[53]:


df.isnull().sum()


# In[ ]:


#merge data country wise


# In[55]:


data_df = df.groupby('Country/Region').sum()


# In[56]:


data_df


# In[65]:


data_df.loc["China"].plot()
data_df.loc["India"].plot()
data_df.loc["US"].plot()
plt.legend()


# In[66]:


data_df.loc['India'].plot()


# In[68]:


data_df.loc['China'][:3].plot()


# In[74]:


countries=list(data_df.index)
max_infection_rates=[]
for c in countries:
    max_infection_rates.append(data_df.loc[c].diff().max())
data_df["max_infection_rates"]=max_infection_rates


# In[75]:


data_df.head()


# In[77]:


data_df=pd.DataFrame(data_df["max_infection_rates"])


# In[81]:


happy_df=pd.read_csv("worldwide_happiness_report.csv")


# In[82]:


happy_df.shape


# In[83]:


happy_df.head()


# In[92]:


cols=["Overall rank","Score","Generosity","Perceptions of corruption"]


# In[93]:


happy_df.drop(cols,axis=1,inplace=True)
happy_df.head()


# In[95]:


happy_df.set_index("Country or region",inplace=True)


# In[98]:


final=data_df.join(happy_df,how="inner")


# In[114]:


corr_matrix=final.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)


# In[102]:


#plot gdp vs maximum infection rate


# In[103]:


x=final["GDP per capita"]
y=final["max_infection_rates"]
sns.scatterplot(x,np.log(y))


# In[104]:


#plot social support vs maximum Infection rate


# In[107]:


x=final["Social support"]
y=final["max_infection_rates"]
sns.scatterplot(x,np.log(y))


# In[108]:


#plot Healthy life expectancy vs maximum Infection rate


# In[109]:


x=final["Healthy life expectancy"]
y=final["max_infection_rates"]
sns.scatterplot(x,np.log(y))


# In[110]:


sns.regplot(x,np.log(y))


# In[111]:


#Plot Freedom to make life choices vs maximum Infection rate¶


# In[112]:


x=final["Freedom to make life choices"]
y=final["max_infection_rates"]
sns.scatterplot(x,np.log(y))


# In[225]:


df_state = pd.read_csv("Total_India_covid-19.csv")


# In[226]:


df_state.shape


# In[227]:


df_state.head()


# In[228]:


#drop unnecessary columns


# In[229]:


df_state.drop(['Latitude','Longitude'],axis=1,inplace=True)


# In[230]:


df_state['Last Updated'] = pd.to_datetime(df_state['Last Updated'], format='%d/%m/%Y %H:%M:%S')


# In[231]:


df_state['year'] = df_state['Last Updated'].dt.year


# In[232]:


df_state.head()


# In[233]:


df_state.drop(['Last Updated'],axis=1,inplace=True)


# In[234]:


df_state.head()


# In[235]:


#let plot the data


# In[236]:


df_state_sorted = df_state.sort_values(by='Active',ascending=False)
df_state_sorted


# In[237]:


plt.figure(figsize=(10,6))
plt.bar(df_state_sorted['State'],df_state_sorted['Active'],color='red')
plt.xlabel('state')
plt.ylabel('active cases')
plt.title('active cases by state')
plt.xticks(rotation=90)
plt.show()


# In[238]:


#what is year analysis for maharastra


# In[ ]:





# In[207]:


df_state_sorted = df_state_sorted['year'] .unique()
df_state_sorted


# In[219]:


#df_state_sorted_corr = df_state_sorted.corr()
#sns.heatmap(df_state_sorted_corr,annot=True,cmap='coolwarm',square=True)


# In[242]:


df_statnew_states = df_state_sorted["State"][:5]


# In[252]:


plt.figure(figsize=(10, 6))  # Adjust figure size as needed

# Scatter plot for two columns
plt.scatter(df_state_sorted['State'][:5], df_state_sorted['Active'][:5], color='skyblue',linestyle = '-',alpha=0.6)

plt.xlabel('State', fontsize=12)  # Customize the x-axis label
plt.plot(df_state_sorted['State'][:5],df_state_sorted['Active'][:5],color='red',linestyle='-',marker='o')
plt.ylabel('Active cases', fontsize=12)  # Customize the y-axis label
plt.title('Scatter Plot of state vs Active', fontsize=14)  # Customize the title
plt.grid(True)  # Add gridlines
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[ ]:


#The active more in maharastra,karnatak,tamilnadu in first three places


# In[253]:


plt.figure(figsize=(10, 6))  # Adjust figure size as needed

# Scatter plot for two columns
plt.scatter(df_state_sorted['State'][:5], df_state_sorted['Deaths'][:5], color='skyblue',linestyle = '-',alpha=0.6)

plt.xlabel('State', fontsize=12)  # Customize the x-axis label
plt.ylabel('death cases', fontsize=12)  # Customize the y-axis label
plt.plot(df_state_sorted['State'][:5],df_state_sorted['Deaths'][:5],color='red',linestyle='-',marker='o')
plt.title('Scatter Plot of state vs death', fontsize=14)  # Customize the title
plt.grid(True)  # Add gridlines
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[254]:


#The deaths more in maharastra,Tamilnadu,karanataka in first three places. Andhra has very less deaths.
#In maharastra 14000 was active cases and close to that was die. It says that Andhra pradhesh has 5000 active cases but only 1% was died.
#That may be tells that Andhra pradesh has good facilities and vaccine centres.More people got vaccine and save their file. 
#may be Andhra prople has more resistance power


# In[ ]:




