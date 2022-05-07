# %%[markdown]
# * Necessary Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import  make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import set_config
from hashlib import sha1
from statsmodels.formula.api import ols
import statsmodels.api as sm 
import warnings
warnings.filterwarnings("ignore")
#%%
import plotly.express as px
import plotly.graph_objects as go
import plotly as py

#%%  ### three dataframe scrapped on three different days. Combining all three files to make final file
df1 = pd.read_json('midtermwork/cardata.json')
df1 = df1.drop_duplicates(subset = ['vin'], keep = 'first')
df2 = pd.read_json('midtermwork/cardata2.json')
df2 = df2.drop_duplicates(subset = ['vin'], keep = 'first')
df3 = pd.read_json('midtermwork/cardata4.json')
df3 = df3.drop_duplicates(subset = ['vin'], keep = 'first')
df_final = pd.concat([df1, df2, df3])

#  %% [markdown]
# * Deleting all the duplicates
# * Adding few extra columns that might help in visualization
df_final = df_final.reset_index(drop=True)
df_final = df_final.drop_duplicates(subset = ['vin'], keep = 'first')
df_final['PriceDiff'] = df_final['msrp'] - df_final['base_price']
df_final['yeargroup'] = df_final['year'].apply(lambda x: '<3 years' if x>2019 else '>3 years')
print(df_final.info())

# %% [markdown]
### Ganesh Exploration
#   * Lets looks at the inventory by make, cylinders and type
#   * Lets look at the Price vs Make cylinders and Type


#%% Inventory lookup
fig, ax = plt.subplots(1,3,figsize=(20,8))
df_final.make.value_counts()[:20].plot(kind= 'bar', title = 'Total Car Inventory', ax = ax[0])
sns.countplot(x = 'cylinders',data = df_final, ax = ax[1]).set_title('Total Invetory by Cylinders')
sns.countplot(x = 'Type',data = df_final,  ax = ax[2]).set_title('Total Invetory by Cylinders')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.show()


# %%[markdown]
# Exploration by Price
# Currently there are of outliers. MSRP is compared for the price less than 80K

fig, ax = plt.subplots(1,3,figsize=(20,8))
sns.boxplot(x='make',y='msrp',data = df_final[(df_final.msrp<80000) & (df_final.msrp>100)],ax=ax[0]).set_title('Price Distribution by Make')
sns.boxplot(x='cylinders',y='msrp',data = df_final[(df_final.cylinders>0) & (df_final.cylinders<9) & (df_final.msrp<80000)& (df_final.msrp>100)],ax=ax[1]).set_title('Price Distribution by Cylinders')
sns.boxplot(x='Type',y='msrp',data = df_final[(df_final.msrp<80000) & (df_final.msrp>100)],ax=ax[2]).set_title('Price Distribution by Type')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.show()
# %% ### interactive plot that will help in visualization
fig = px.scatter(df_final[(df_final.msrp<80000) & (df_final.msrp>100) & (df_final.base_price<100000)], 
                x = 'base_price',
                y= 'msrp',color = 'make', 
                hover_name = 'make', 
                hover_data = ['cylinders', 'trasmission', 'Type', 'mileage'],
                size = 'mileage', size_max = 30, 
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

# %% [markdown]
### Xuan Exploration
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
    sns.barplot(x='fuelType', y='msrp', data=Selected[Selected.make == brand], ax=ax[1], estimator=np.median, ci=False).set_title(f'Distribution of {brand} by Type')
    # Draw a scatterplot of the relationship between mileage and base_price, the color is determined by Type, The shape of the point is determined by yeargroup, and the transparency of the point is 0.6
    sns.scatterplot(x='mileage', y='base_price', data=Selected[Selected.make == brand], hue='Type', ax=ax[2], style='yeargroup', alpha=0.6).set_title(f'Mileage vs Base_price for {brand}')
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

#%%
#Kwan Hang exploration
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
fig, ax = plt.subplots(3,figsize=(20,30), dpi=500)
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

# %% [markdown]
### Ganesh Machine Learning

# %%  segregattign the data for the mdoel building
data = df_final[df_final.groupby('make')['make'].transform('size') > 1000].drop([ 'stock_Number', 'vin', 'storeName', 'city', 'state', 'exteriorColor', 'interiorColor', 'storezip', 'PriceDiff', 'yeargroup'], axis=1)
data = data.loc[(data.msrp<80000) & (data.msrp>2000)]
### further subsetting the data with the models that more more than 20 cars in inventory
data.dropna(inplace=True)
data = data.reset_index(drop=True)
### plotting the correlation plot
plt.figure(figsize=(16,8))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.triu(np.ones_like(data.corr(), dtype=bool))
sns.heatmap(data.corr(),mask = mask,  annot = True)

### taking away the baseprice as it has really high correlation and there is no way for new customer to know it. 
### separating categorical and continious columns. All cols will be used in subsetting new data to feed into the model
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

for col in data.columns:
    if data[col].nunique()<30:
        print(col)
        print('\n')
        print(data[col].value_counts())
        print('\n')


# %%  ### instancing the model
lm = LinearRegression()

# set up preprocessing for categorical columns
imp_constant = SimpleImputer(strategy='constant')
ohe = OneHotEncoder(handle_unknown='ignore')

# set up preprocessing for numeric columns
imp_median = SimpleImputer(strategy='median', add_indicator=True)

### creating the tranformer with pipline in it. 
preprocess = make_column_transformer(
                                    (make_pipeline(imp_constant, ohe), catcols),
                                    (imp_median, Numcols),
                                   )


# %%
pipeline = make_pipeline(preprocess, lm)
# %%  ### getting the r square and the rmse for the pipeline
print(cross_val_score(pipeline, X, y, cv = 10, scoring= 'neg_root_mean_squared_error').mean())
print(cross_val_score(pipeline, X, y, cv = 10, scoring= 'r2').mean())

pipeline.fit(X, y)

# %% [markdown]

#### Model Evaluation   
make_trained = X.make.unique()   ### getting the list of the unique car make used to make model
test_data = pd.read_json('midtermwork/newdata.json')  ### reading new file
test_data = test_data.drop_duplicates(subset = 'vin')   ### dropping duplicate files
test_data = test_data[test_data['make'].isin(make_trained)]  ### ensuring new testing data has only make from trained model
test_data = test_data[~test_data['vin'].isin(df_final.vin.unique())]   ### ensuring new data doesn't have the 
test_data = test_data[(test_data.msrp<80000)  & (test_data.msrp>2000)]   ### ensuing we predict on the price range between 2000 and 8000
print(test_data.head())
print(test_data.shape[0])
# %%  ### getting the predicted price by the mdoel
predicted_price = pipeline.predict(test_data[Allcols])
# %%
test_data['predicted_price'] = predicted_price
test_data['predicted_diff'] = test_data['predicted_price'] - test_data['msrp']

# %%   ### plotting the predicted vs acutal price 
plt.figure(figsize=(20,16))
fig = px.scatter(test_data,
                x = 'msrp',
                y= 'predicted_price',color = 'make', 
                hover_name = 'make', 
                hover_data = ['cylinders', 'trasmission', 'Type', 'mileage', 'year'],trendline= 'ols', 
                width = 800, height = 500
                )
fig.update_layout(title_text='Predicted vs Acutal', title_x=0.5)
# %%  ### Residual plot
sns.residplot('msrp','predicted_price',  test_data)
# %%
# set_config(display = 'diagram')
# pipeline
# %%
from sklearn.metrics import r2_score, mean_squared_error
# %%
print('Printing the r2_score and rmse in the new data')
print(r2_score(test_data.msrp, test_data.predicted_price))
print(np.sqrt(mean_squared_error(test_data.msrp, test_data.predicted_price)))

### github source for the below code is : https://johaupt.github.io/blog/columnTransformer_feature_names.html
import sklearn
import warnings
def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names
# %%  ### making the dataframe for the coeff with adjacent columns
coef = pd.DataFrame(pipeline.named_steps['linearregression'].coef_.flatten(), index=get_feature_names(preprocess))
coef = coef.reset_index().sort_values(by = [0])
 ### getting the top and last 50 columns with the linear model coefficient
topandlast50 = pd.concat([coef.head(50), coef.tail(50)]).reset_index(drop=True)
# %%  ### plotting the model coefficient
topandlast50.plot.barh(x='index', y =0, figsize = (10,20))

# %%
#Xuan linear reggression

#%%
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

# %%

# %%
