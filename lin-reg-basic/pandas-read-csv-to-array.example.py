import pandas as pd
df=pd.read_csv('kc_house_data.csv', sep=',',header=None)
df.values


new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = new_header #set the header row as the df header



print(df.head())

price_array = df['price']
print(price_array)

# Drop id and date  columns
df.drop(df.columns[0:2], axis=1, inplace=True)
#df.drop(df.columns[0], axis=1, inplace=True)
#print(price_array)
print(df.head())
# ,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15
#df.drop('floors','waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement')
#df.drop('yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'yr_built')
df.drop(df.columns[5:], axis=1, inplace=True)


print(df.head())

