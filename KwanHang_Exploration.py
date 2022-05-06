#%%
from hashlib import sha1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
from statsmodels.formula.api import ols
import statsmodels.api as sm 
#%%
import plotly as py
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
df_final = pd.concat([df1,df2, df3])

# %%   ### Deleting all the duplicates
## Adding few extra columns
df_final = df_final.drop_duplicates(subset = ['vin'], keep = 'first')
df_final['PriceDiff'] = df_final['msrp'] - df_final['base_price']
df_final['yeargroup'] = df_final['year'].apply(lambda x: '<3 years' if x>2019 else '>3 years')
# %%
df_final.info()

# %%
#Daniel exploration
fig, ax = plt.subplots(1,3,figsize=(20,16))
sns.boxplot(x='state',y='msrp',data = df_final[(df_final.msrp<80000) & (df_final.msrp>100)],ax=ax[0]).set_title('Price Distribution by State')
sns.boxplot(x='exteriorColor',y='msrp',data = df_final[(df_final.msrp<80000)& (df_final.msrp>100)],ax=ax[1]).set_title('Price Distribution by ExtColor')
sns.boxplot(x='engineType',y='msrp',data = df_final[(df_final.msrp<80000) & (df_final.msrp>100)],ax=ax[2]).set_title('Price Distribution by Engine Type')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig("Boxplot of different prices across states, exterior colors and engine Type.png")
plt.show()
# %%
sns.distplot(df_final.msrp)
plt.savefig("price distribution.png")
plt.show()
# %%
plt.title('Car Price Spread')
sns.boxplot(y=df_final.msrp)
plt.savefig("Price distribution box plot.png")
plt.show()
#%%
plt.scatter(df_final['make'], df_final['msrp'])
plt.grid()
plt.savefig("Price distribution across makes scatterplot.png")
plt.xticks(rotation=90)
fig.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(2,2,figsize=(20,16), dpi=500)
ax = ax.ravel()
sns.scatterplot(x = 'horsepower' , y = 'msrp', data = df_final, ax = ax[0] ).set_title('Price distribution by horsepower')
sns.scatterplot(x = 'engineTorque' , y = 'msrp', data = df_final, ax = ax[1] ).set_title('Price distribution by engine torque')
sns.scatterplot(x='horsepowerRpm', y='msrp', data=df_final, ax=ax[2]).set_title('Price Distribution by horsepowerRpm')
sns.scatterplot(x='engineTorqueRpm', y='msrp', data=df_final, ax=ax[3]).set_title('Price Distribution by Engine Torque Rpm')
plt.savefig("Scatterplot of msrp distribution over 4 different variables.png")
plt.show()
# %%
fig, ax = plt.subplots(3,figsize=(20,16), dpi=500)
ax = ax.ravel()
sns.boxplot(x='make', y='msrp', data=df_final, ax=ax[0]).set_title('Price Distribution by Make')
sns.boxplot(x='engineSize', y='msrp', data=df_final, ax=ax[1]).set_title('Price Distribution by engineSize')
sns.boxplot(x='make', y='PriceDiff', data=df_final, ax=ax[2]).set_title('Base price and msrp differences across makes')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig("Boxplot of 3 more metrics of measure.png")
plt.show()
# %%
sns.scatterplot(x = 'horsepower' , y = 'horsepowerRpm', data = df_final).set_title('correlation of horsepower vs horsepoweraRpm')
plt.savefig("correlation between horsepower and horsepowerRpm.png")
plt.show()
#%%
sns.scatterplot(x = 'engineTorque' , y = 'engineTorqueRpm', data = df_final).set_title('correlation of enginetorque vs enginetorqueRpm')
plt.savefig("correlation between enginetorque and enginetorqueRpm.png")
plt.show()
# %%
# Linear regression model using least squares, with msrp as dependant variable and 7 other independant variables
data = df_final.dropna(axis=0, subset=["msrp"])
modelmsrp = ols(formula="msrp ~ C(make)+ C(model) + horsepower + horsepowerRpm + engineTorque + engineTorqueRpm + engineSize", data=data)
modelmsrpfit = modelmsrp.fit()
print(modelmsrpfit.summary())

#%%
#Results interpretation
import numpy as np
print(np.exp(modelmsrpfit.params))
print(np.exp(modelmsrpfit.conf_int()))

# %%
