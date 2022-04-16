#%%
from hashlib import sha1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

#%%
df1 = pd.read_json('cardata.json')
df1 = df1.drop_duplicates(subset = ['vin'], keep = 'first')
df2 = pd.read_json('cardata2.json')
df2 = df2.drop_duplicates(subset = ['vin'], keep = 'first')
#%%
df3 = pd.read_json('cardata4.json')
df3 = df3.drop_duplicates(subset = ['vin'], keep = 'first')

# %%
df_final = pd.concat([df1,df2, df2])

# %%   ### Deleting all the duplicates
## Adding few extra columns
df_final = df_final.drop_duplicates(subset = ['vin'], keep = 'first')
df_final['PriceDiff'] = df_final['msrp'] - df_final['base_price']
df_final['yeargroup'] = df_final['year'].apply(lambda x: '<3 years' if x>2019 else '>3 years')
# %%
df_final.info()
# %% [markdown]
### Exploration
#   * Lets looks at the inventory by make, cylinders and type
#   * Lets look at the Price vs Make cylinders and Type


#%% Inventory lookup
fig, ax = plt.subplots(1,3,figsize=(20,16))
df_final.make.value_counts()[:20].plot(kind= 'bar', title = 'Total Car Inventory', ax = ax[0])
sns.countplot(x = 'cylinders',data = df_final, ax = ax[1]).set_title('Total Invetory by Cylinders')
sns.countplot(x = 'Type',data = df_final,  ax = ax[2]).set_title('Total Invetory by Cylinders')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.show()


# %% 
# Exploration by Price
# Currently there are of outliers. MSRP is compared for the price less than 80K

fig, ax = plt.subplots(1,3,figsize=(20,16))
sns.boxplot(x='make',y='msrp',data = df_final[(df_final.msrp<80000) & (df_final.msrp>100)],ax=ax[0]).set_title('Price Distribution by Make')
sns.boxplot(x='cylinders',y='msrp',data = df_final[(df_final.cylinders>0) & (df_final.cylinders<9) & (df_final.msrp<80000)& (df_final.msrp>100)],ax=ax[1]).set_title('Price Distribution by Cylinders')
sns.boxplot(x='Type',y='msrp',data = df_final[(df_final.msrp<80000) & (df_final.msrp>100)],ax=ax[2]).set_title('Price Distribution by Type')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.show()
# %%
# %%
fig = px.scatter(df_final[(df_final.msrp<80000) & (df_final.msrp>100) & (df_final.base_price<100000)], 
                x = 'base_price',
                y= 'msrp',color = 'make', 
                hover_name = 'make', 
                hover_data = ['cylinders', 'trasmission', 'Type', 'mileage'],
                size = 'mileage', size_max = 20, 
                )
fig.update_layout(title_text='Baseprice vs MSRP', title_x=0.5)

fig.show()
# %% [markdown]
#### Looking into car brands with highest selection

Brands = ['Toyota','Chevrolet','Ford','Honda' ]
Selected = df_final.loc[((df_final.make=='Toyota') | (df_final.make=='Chevrolet') | (df_final.make =='Ford') | (df_final.make =='Honda')) & (df_final.msrp<70000) & (df_final.msrp>1000)]

for brand in Brands:
    fig, ax = plt.subplots(1, 3, figsize = (20,10))
    fig.suptitle(f'Overall Breakdown of {brand}')
    Selected.loc[Selected.make==brand].model.value_counts()[:10].plot(kind = 'bar',ax=ax[1], title = f'Top 10 Models for {brand} By Inventory')
    sns.barplot(x = 'fuelType', y = 'msrp', hue = 'Type', data = Selected[Selected.make==brand], ax = ax[0], estimator= np.median, ci = False).set_title(f'Distribution of {brand} by Type')
    sns.scatterplot(x = 'mileage' , y = 'msrp', data = Selected[Selected.make==brand], hue = 'yeargroup', ax = ax[2], style = 'Type', alpha= 0.6 ).set_title(f'Mileage vs MSRP for {brand}')
    plt.figure(plt.tight_layout())


# %%

# %%
