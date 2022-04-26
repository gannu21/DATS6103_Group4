#%%
from hashlib import sha1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly as py

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
df_final = df_final.reset_index(drop=True)
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
# Select 'make', 'cylinder', 'exteriorColor', 'interiorColor', 'engineType', 'model' and 'base_price'.
fig, ax = plt.subplots(2,3,figsize=(20,16), dpi=500)
ax = ax.ravel()
# Draw the top 20 makes by quantity
df_final.make.value_counts()[:20].plot(kind= 'bar', title = 'Total Car Inventory by Make', ax = ax[0])
# Draw the cylinder population
sns.countplot(x = 'cylinders',data = df_final, ax = ax[1]).set_title('Total Invetory by Cylinders')
# Draw the exteriorColor population
sns.countplot(x = 'exteriorColor', data = df_final, ax = ax[2]).set_title('Total Inventory by exteriorColor')
# Draw the interiorColor population
sns.countplot(x = 'interiorColor', data = df_final, ax = ax[3]).set_title('Total Inventory by interiorColor')
# Draw the engineType population
sns.countplot(x = 'engineType', data = df_final, ax = ax[4]).set_title('Total Inventory by engineType')
# Draw the model population
df_final.model.value_counts()[:20].plot(kind='bar', title='Total Car Inventory by model', ax=ax[5])

# x-axis labels rotated 90°
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig("Histogram of the six metrics.png")
plt.show()

# %% 
# Exploration by Price
# Currently there are of outliers. MSRP is compared for the price less than 80K

# Draw boxplots of the above six metrics
fig, ax = plt.subplots(2,3,figsize=(20,16), dpi=500)
ax = ax.ravel()
sns.boxplot(x='make', y='base_price', data=df_final, ax=ax[0]).set_title('Price Distribution by Make')
sns.boxplot(x='cylinders', y='base_price', data=df_final, ax=ax[1]).set_title('Price Distribution by Cylinders')
sns.boxplot(x='exteriorColor', y='base_price', data=df_final, ax=ax[2]).set_title('Price Distribution by Exterior Color')
sns.boxplot(x='interiorColor', y='base_price', data=df_final, ax=ax[3]).set_title('Price Distribution by Interior Color')
sns.boxplot(x='engineType', y='base_price', data=df_final, ax=ax[4]).set_title('Price Distribution by Engine Type')
model_20 = df_final[df_final.model.isin(df_final.model.value_counts()[:20].index)]
sns.boxplot(x='model', y='base_price', data=model_20, ax=ax[5]).set_title('Price Distribution by model')

#sns.boxplot(x='make',y='msrp',data = df_final[(df_final.msrp<80000) & (df_final.msrp>100)],ax=ax[0]).set_title('Price Distribution by Make')
#sns.boxplot(x='cylinders',y='msrp',data = df_final[(df_final.cylinders>0) & (df_final.cylinders<9) & (df_final.msrp<80000)& (df_final.msrp>100)],ax=ax[1]).set_title('Price Distribution by Cylinders')
#sns.boxplot(x='Type',y='msrp',data = df_final[(df_final.msrp<80000) & (df_final.msrp>100)],ax=ax[2]).set_title('Price Distribution by Type')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig("Boxplot of the six metrics.png")
plt.show()

# %%
# %%
# Draw a scatterplot of the relationship between base_price (the minimum used car price) and the msrp (manufacturer's recommended price). Color is determined by "make" and size is determined by "mileage", up to 20.
fig = px.scatter(df_final[(df_final.msrp!=0) & (df_final.base_price < 150000)], 
                x = 'base_price',
                y= 'msrp',color = 'make', 
                hover_name = 'make', 
                hover_data = ['cylinders', 'trasmission', 'Type', 'mileage'],
                size = 'mileage', size_max = 20, 
                )
fig.update_layout(title_text='Baseprice vs MSRP', title_x=0.5)
py.offline.plot(fig, filename="test.html")
fig.show()

# %% [markdown]
#### Looking into car brands with highest selection

# Select the top 5 makes by quantity
Brands = df_final.make.value_counts()[:5].index
Selected = df_final[df_final.make.isin(Brands)]

for brand in Brands:
    # Analyzed by make(brand), the number of brands is euqal to the length of variables Brands
    fig, ax = plt.subplots(1, 3, figsize=(20, 10), dpi=500)
    fig.suptitle(f'Overall Breakdown of {brand}')
    # Barplot of the top ten models
    Selected.loc[Selected.make == brand].model.value_counts()[:10].plot(kind='bar', ax=ax[0], title=f'Top 10 Models for {brand} By Inventory')
    # Draw a barplot of the relationship between fuel type and mrsp, the color is determined by Type
    sns.barplot(x='fuelType', y='msrp', hue='Type', data=Selected[Selected.make == brand], ax=ax[1], estimator=np.median, ci=False).set_title(f'Distribution of {brand} by Type')
    # Draw a scatterplot of the relationship between mileage and base_price, the color is determined by Type, The shape of the point is determined by yeargroup, and the transparency of the point is 0.6
    sns.scatterplot(x='mileage', y='base_price', data=Selected[Selected.make == brand], hue='Type', ax=ax[2], style='yeargroup', alpha=0.6).set_title(f'Mileage vs MSRP for {brand}')
    plt.tight_layout()
    plt.savefig(f"Plots of metrics-{brand}.png")
plt.show()


# %%
plt.scatter(np.log(df_final['base_price']), df_final['cylinders'])
plt.savefig("The relationship between base_price and cylinders.png")
plt.show()

# %%
# Find the correlation coefficient matrix heat map (only for numerical data, the numbers in the square represent the size of the covariance)
# Set palette
plt.figure(dpi=500)
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma = 0.8, as_cmap = True)
ax = sns.heatmap(df_final.corr(), vmax = 1, cmap = cmap, annot = True, annot_kws={'size':5, 'weight':'bold'})
plt.tight_layout()
plt.savefig("The correlation coefficient matrix heat map.png")
plt.show()

# %%
# Do Linear regression using statsmodels from the last assignment.
from statsmodels.formula.api import ols

# Preprocessing, the general idea is to remove the situation with a small amount of data under a certain feature.
# Because there are too many models, the first 20 are screened, and the accuracy of my test: 30<10<20=25
data = df_final[df_final.model.isin(df_final.model.value_counts()[:20].index)]
# The car mumbers of prior three cylinders largely beyond others, so the number of cylinders in the top three is selected.
data = data[data.cylinders.isin(data.cylinders.value_counts()[:3].index)]
# Delete rows where mrsp is nan
data = data.dropna(axis=0, subset=["msrp"])
# Because most of the engineTypes are Gas, so filter Gas
data = data[data.engineType == "Gas"]
# Filter exteriorColor
data = data[data.exteriorColor.isin(data.interiorColor.value_counts()[:6].index)]
# Filter interiorColor
data = data[data.interiorColor.isin(data.interiorColor.value_counts()[:3].index)]

# Divide the train set and the test set, the train set is 70%, and the test set is 30%.
# The latter random_state is the random number seed. When set to the same value, the train set and the test set obtained by random sampling are the same each time.
train = data.sample(frac=0.7, random_state=1234).copy()
test = data[~data.index.isin(train.index)].copy().reset_index(drop=True)
train = train.reset_index(drop = True)
# Build a least squares regression model with base_price as the dependent variable.
# After ~ is an independent variable, and adding C means that the variable is a categorical variable.
modelBasePrice = ols(
    formula="base_price ~ C(make) + msrp + C(model) + C(cylinders) + C(engineType) + C(interiorColor) + C(exteriorColor)",
    data=train
)

modelBasePriceFit = modelBasePrice.fit()
print(modelBasePriceFit.summary())

# %%
# Put the base_price of the test set into the 'true_values' column.
modelPredictions = pd.DataFrame(test.base_price.values, columns=['true_values'])
# Put predicted values into 'test_values'.
modelPredictions['predict_values'] = modelBasePriceFit.predict(test)
# Difference between predicted value and actual value
modelPredictions['diff'] = abs(modelPredictions['predict_values'] - modelPredictions['true_values'])
# The ratio of the difference to the true value
modelPredictions['diff_percent(%)'] = abs((modelPredictions['predict_values'] - modelPredictions['true_values'])/modelPredictions['true_values']) * 100
# Sort the table by ratio from smallest to largest
modelPredictions = modelPredictions.sort_values(by="diff_percent(%)")
# Update index
modelPredictions = modelPredictions.reset_index(drop=True)

# %%
# Predict results.
# The situation that the error rate is less than 5%.
percents = [5, 10, 20, 30]
for percent in percents:
    rate = len(modelPredictions[modelPredictions['diff_percent(%)'] <= percent]) / len(modelPredictions)
    print(f"The rate of prediction accuracy greater than {100-percent}% is: {rate}")

# %%
# Judging the significant impact of features.
print(modelBasePriceFit.params)

# %%
# Calculate the Car Types that can be purchased under a certain budget, and just filter directly.
money = 300000 
# 300000 is just an example.
print(df_final[df_final.base_price < money].Type.value_counts().index)
# %%
