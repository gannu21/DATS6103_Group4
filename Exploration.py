#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn import set_config


#%%
df1 = pd.read_json('cardata.json')
df1 = df1.drop_duplicates(subset = ['vin'], keep = 'first')
df2 = pd.read_json('cardata2.json')
df2 = df2.drop_duplicates(subset = ['vin'], keep = 'first')
#%%
df3 = pd.read_json('cardata4.json')
df3 = df3.drop_duplicates(subset = ['vin'], keep = 'first')

# %%
df_final = pd.concat([df1,df2, df3])

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
    # Barplot of the top ten models
    Selected.loc[Selected.make == brand].model.value_counts()[:10].plot(kind='bar', ax=ax[0], title=f'Top 10 Models of {brand} By Inventory')
    # Draw a barplot of the relationship between fuel type and mrsp, the color is determined by Type
    sns.barplot(x='fuelType', y='msrp', data=Selected[Selected.make == brand], ax=ax[1], estimator=np.median, ci=False).set_title(f'MSRP Distribution of {brand} by FuelType')
    # Draw a scatterplot of the relationship between mileage and base_price, the color is determined by Type, The shape of the point is determined by yeargroup, and the transparency of the point is 0.6
    sns.scatterplot(x='mileage', y='base_price', data=Selected[Selected.make == brand], hue='Type', ax=ax[2], style='yeargroup', alpha=0.6).set_title(f'Mileage vs Base_price of {brand}')
    plt.tight_layout()
    plt.savefig(f"Plots of metrics-{brand}.png")
plt.show()


# %%

# %% [markdown]
### Machine Learning
# * For machine learning we decided to do modelling just on the Toyota brands. We have arround 2000 data points for toyota. 

# %%
data = df_final[df_final.groupby('make')['make'].transform('size') > 1000].drop([ 'stock_Number', 'vin', 'storeName', 'city', 'state', 'exteriorColor', 'interiorColor', 'storezip', 'PriceDiff', 'yeargroup'], axis=1)
data = data.loc[(data.msrp<80000) & (data.msrp>2000)]
# %%
### further subsetting the data with the models that more more than 20 cars in inventory
data.dropna(inplace=True)
data = data.reset_index(drop=True)

#%%
plt.figure(figsize=(16,8))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.triu(np.ones_like(data.corr(), dtype=bool))
sns.heatmap(data.corr(),mask = mask,  annot = True)


### take away the baseprice as it has really high correlation and there is no way for new customer to know it. 
#%%

catcols = [ 'make', 'model','body', 'trasmission','engineType','fuelType', 'Type', 'year', 'mpgCity', 'mpgHighway','cylinders','horsepowerRpm', 'engineSize', 'engineTorque', 'engineTorqueRpm' ]
print(catcols)
print()
Numcols = ['mileage', 'horsepower']
print(Numcols)
print()
print(f'total columns in data : {data.shape[1]} \n total cols used : {len(catcols)+len(Numcols)}')
Allcols = catcols + Numcols
print('\n')
print(Allcols)
print(len(Allcols))



X = data[Allcols]
y = data['msrp']


#%%
for col in data.columns:
    if data[col].nunique()<30:
        print(col)
        print('\n')
        print(data[col].value_counts())
        print('\n')

# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import  make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
# %%

# %%
lm = LinearRegression()

# set up preprocessing for categorical columns
imp_constant = SimpleImputer(strategy='constant')
ohe = OneHotEncoder(handle_unknown='ignore')

# set up preprocessing for numeric columns
imp_median = SimpleImputer(strategy='median', add_indicator=True)



preprocess = make_column_transformer(
                                    (make_pipeline(imp_constant, ohe), catcols),
                                    (imp_median, Numcols),
                                   )


# %%
pipeline = make_pipeline(preprocess, lm)
# %%
print(cross_val_score(pipeline, X, y, cv = 10, scoring= 'neg_root_mean_squared_error').mean())
print(cross_val_score(pipeline, X, y, cv = 10, scoring= 'r2').mean())

pipeline.fit(X, y)

# %% [markdown]

#### Model Evaluation
make_trained = X.make.unique()   ### getting the list of the unique car make used to make model
test_data = pd.read_json('newdata.json')  ### reading new file
test_data = test_data.drop_duplicates(subset = 'vin')   ### dropping duplicate files
test_data = test_data[test_data['make'].isin(make_trained)]  ### ensuring new testing data has only make from trained model
test_data = test_data[~test_data['vin'].isin(df_final.vin.unique())]   ### ensuring new data doesn't have the 
test_data = test_data[(test_data.msrp<80000)  & (test_data.msrp>2000)]
print(test_data.head())
print(test_data.shape[0])
# %%
predicted_price = pipeline.predict(test_data[Allcols])
# %%
test_data['predicted_price'] = predicted_price
test_data['predicted_diff'] = test_data['predicted_price'] - test_data['msrp']

# %%
plt.figure(figsize=(20,16))
fig = px.scatter(test_data,
                x = 'msrp',
                y= 'predicted_price',color = 'make', 
                hover_name = 'make', 
                hover_data = ['cylinders', 'trasmission', 'Type', 'mileage', 'year'],trendline= 'ols', 
                width = 1200, height = 600
                )
fig.update_layout(title_text='Predicted vs Acutal', title_x=0.5)
# %%
sns.residplot('msrp','predicted_price',  test_data)
# %%
set_config(display = 'diagram')
pipeline
# %%
