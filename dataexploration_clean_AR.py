# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# %% Data Preprocessing
#red in data from json file
cars = pd.read_json('cardata.json', orient='columns')
cars = cars.drop_duplicates(subset=None, keep='first', inplace=False)
# %% Data Summary
print(cars.head())
print(cars.tail())
print(cars.info())
print(cars.shape)

# %% Summary Statistics
cars.describe()

# %%
#plot formatting
sns.set(style="white", palette="muted", color_codes=True,rc={'figure.figsize':(11.7,8.27)})
rs = np.random.RandomState(10)

# %% SMART QUESTION 1 - COLOR

print(cars["exteriorColor"].nunique())

print(cars["interiorColor"].nunique())

#%% color frequency
int_colors = ["#050505","#7B8085", "#F5D60C",  "#F5E6D3", "#473A2A", "#E8E5E1", "#FF9814","#54946C", "#C72C36","#2C53C7"]

cars['excolor_freq'] = cars['exteriorColor'].map(cars['exteriorColor'].value_counts())

cars.sort_values('excolor_freq', inplace=True)

sns.histplot(data=cars, x="exteriorColor",multiple="stack", hue="interiorColor", palette = sns.color_palette(int_colors,10))
plt.title("Frequency of Color",fontdict= { 'fontsize': 14, 'fontweight':'bold'})
plt.xticks(rotation=45)
plt.xlabel('Exterior Color')
plt.ylabel('Number of Cars in Stock')
#We have 2 colors to consider, interior and exterior. Let's create a pivot table and heatmap to see see how these combinations affect price.
# %%
pivot_q2 =  np.round(pd.pivot_table(cars, values='msrp',
                                index = ['exteriorColor'],
                                columns=[ 'interiorColor'],
                                aggfunc=np.mean,
                                fill_value=0),2)

print(pivot_q2)

#Lets add in a heat map to make this easier to read



sns.heatmap(pivot_q2,fmt='g', center = 30000, vmin = -1000, cmap = "rocket_r")
plt.title("Average MSRP for Interior and Exterior Color Combinations",fontdict= { 'fontsize': 14, 'fontweight':'bold'})
plt.xlabel('Interior Color')
plt.ylabel('Exterior Color')

#Statistical Test

import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('msrp ~ C(exteriorColor)', data=cars).fit()
model2 = ols('msrp ~ C(interiorColor)', data=cars).fit()

import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

normality_plot, stat = stats.probplot(model2.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

from scipy.stats import bartlett
stat, p = bartlett(cars[cars['exteriorColor'] == 'Yellow']['msrp'],cars[cars['exteriorColor'] == 'Purple']['msrp'], cars[cars['exteriorColor'] == 'Orange']['msrp'], cars[cars['exteriorColor'] == 'Tan']['msrp'], cars[cars['exteriorColor'] == 'Gold']['msrp'], cars[cars['exteriorColor'] == 'Green']['msrp'], cars[cars['exteriorColor'] == 'Brown']['msrp'], cars[cars['exteriorColor'] == 'Red']['msrp'], cars[cars['exteriorColor'] == 'Blue']['msrp'], cars[cars['exteriorColor'] == 'Silver']['msrp'], cars[cars['exteriorColor'] == 'Gray']['msrp'], cars[cars['exteriorColor'] == 'Black']['msrp'],cars[cars['exteriorColor'] == 'White']['msrp'])
print(p)

stat, p = bartlett(cars[cars['interiorColor'] == 'Black']['msrp'],cars[cars['interiorColor'] == 'Gray']['msrp'], cars[cars['interiorColor'] == 'Yellow']['msrp'], cars[cars['interiorColor'] == 'Tan']['msrp'], cars[cars['interiorColor'] == 'Brwon']['msrp'], cars[cars['interiorColor'] == 'White']['msrp'], cars[cars['interiorColor'] == 'Orange']['msrp'], cars[cars['interiorColor'] == 'Green']['msrp'], cars[cars['interiorColor'] == 'Red']['msrp'], cars[cars['interiorColor'] == 'Blue']['msrp'])
print(p)


#Failed Assumptions Try Welch's

import pingouin as pg

pg.welch_anova(dv='msrp', between='exteriorColor', data=cars)

pg.welch_anova(dv='msrp', between='interiorColor', data=cars)

#post hoc

eC = pg.pairwise_gameshowell(dv='msrp', between='exteriorColor', data=cars)

eC.to_csv (r'C:\Users\15408\Documents\GitHub\gannu21-DATS6103_Group4\pwgs_eC.csv', index = False)

iC = pg.pairwise_gameshowell(dv='msrp', between='interiorColor', data=cars)
iC.to_csv (r'C:\Users\15408\Documents\GitHub\gannu21-DATS6103_Group4\pwgs_iC.csv', index = False)

#%% SMART QUESTION 2 - TYPE

print(cars["Type"].nunique())
print(cars["Type"].unique())

cars['average_base_price_type'] = cars.groupby(['Type'])['msrp'].transform('mean')
cars.sort_values('average_base_price_type', inplace=True)

cars_msrp_mean = cars['msrp'].mean()

sns.histplot(data=cars, x="Type", color = "#f00a11")
plt.title("Frequency of Types",fontdict= { 'fontsize': 14, 'fontweight':'bold'})
plt.xticks(rotation=45)
plt.xlabel('Type')
plt.ylabel('Number of Cars in Stock')

sns.boxplot(data = cars, x = 'Type', y = 'msrp', showfliers=False, palette = "RdYlGn")
plt.title("Average MSRP by Type",fontdict= { 'fontsize': 14, 'fontweight':'bold'})
plt.xticks(rotation=45)
plt.xlabel('Type')
plt.ylabel('MSRP')
plt.axhline(y = cars_msrp_mean, color = '#18a2ed',linestyle = 'dashed', label = "Avg MSRP", linewidth =2)
plt.legend( loc = 'upper right')

pg.welch_anova(dv='msrp', between='Type', data=cars)

Tp = pg.pairwise_gameshowell(dv='msrp', between='Type', data=cars)
Tp.to_csv (r'C:\Users\15408\Documents\GitHub\gannu21-DATS6103_Group4\pwgs_Tp.csv', index = False)

# %% SMART QUESTION 3 - MAKE

cars["make_model"] = cars["make"] + " " + cars["model"]

print(cars["make_model"].nunique())
print(cars["make"].nunique())
print(cars["model"].nunique())

#Number of make and model
print(cars["make_model"].nunique())

sns.histplot(data=cars, x="make_model", hue="Type")

# %%
#Lets pull the top make/model combos available where stock is within top 25 percentile to visualize

cars['make_model_freq'] = cars['make_model'].map(cars['make_model'].value_counts())

cars['percentile_freq'] = cars.make_model_freq.rank(pct = True)

#subset data fr top 25 %
carsTop = cars.loc[cars['percentile_freq'] > .75]

#try plotting again
carsTop.sort_values('make_model_freq', inplace=True)
#%%

colors_edit = ['#c41e27', '#de402e', '#f16640', '#f98e52', '#fdb567',   '#abdb6d', '#84ca66', '#5ab760', '#2aa054', '#0f8446']


sns.histplot(data=carsTop, x="make_model", hue="Type", multiple="stack", palette = sns.color_palette(colors_edit, 9))
plt.title("Frequency of Make and Models",fontdict= { 'fontsize': 14, 'fontweight':'bold'})
plt.xticks(rotation=45)
plt.xlabel('Make and Model')
plt.ylabel('Number of Cars in Stock')

carsTop2 = cars.loc[cars['percentile_freq'] > .0]

#We will add in Type with make and model
pivot_heat= np.round(pd.pivot_table(carsTop2, values='msrp',
                                index = ["make"],
                                columns = ['year'],
                                aggfunc=np.mean,
                                fill_value=0),2)

print(pivot_heat)

#Lets add in a heat map to make this easier to read

sns.heatmap(pivot_heat, fmt='.3F', center = 65000, vmin = 0, cmap = "rocket_r")
plt.title("Average MSRP for Make by Year",fontdict= { 'fontsize': 14, 'fontweight':'bold'})
plt.xlabel('Year')
plt.ylabel('Make')


pg.welch_anova(dv='msrp', between='make', data=cars)
Mk = pg.pairwise_gameshowell(dv='msrp', between='make', data=cars)
Mk.to_csv (r'C:\Users\15408\Documents\GitHub\gannu21-DATS6103_Group4\pwgs_Mk.csv', index = False)
