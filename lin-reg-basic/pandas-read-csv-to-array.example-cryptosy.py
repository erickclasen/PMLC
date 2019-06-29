import pandas as pd
df=pd.read_csv('cbpro_crypto_price_volume_file.csv', sep=',',header=None)
df.values


#new_header = df.iloc[0] #grab the first row for the header
#df = df[1:] #take the data less the header row
#df.columns = new_header #set the header row as the df header
df.columns = ['date','BTC-USD','BCH-USD','ETC-USD','ETH-USD','LTC-USD','BTC-VOL','BCH-VOL','ETC-VOL','ETH-VOL','LTC-VOL']


print(df.head())


