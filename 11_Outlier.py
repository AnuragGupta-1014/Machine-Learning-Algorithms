import pandas as pd 

# df = pd.read_csv('heights.csv')

# print(df['height'].head(4))

# max_thresold = df['height'].quantile(0.95)      # above the output is consider as outlier
# min_thresold = df['height'].quantile(0.05)      # under the output is consider as outlier
# print(max_thresold)
# print(min_thresold)

# print(df[df['height']>max_thresold])        #this will give outlier 
# print(df[df['height']<min_thresold])        #this will give outlier 


# # removing it simply

# New_df = df[(df['height'] >min_thresold) & (df['height'] < max_thresold)]
# print(New_df)



# Now work on coplex data

dff = pd.read_csv('bhp.csv')
print(dff.head())
# print(dff.price_per_sqft)
min_thresold,max_thresold = dff.price_per_sqft.quantile([0.001,0.999])
print(min_thresold,max_thresold)                  # this are the outlier

# remove as same as above
New_dff = dff[(dff['price_per_sqft'] >min_thresold) & (dff['price_per_sqft'] < max_thresold)]

print(dff.shape)              # before outlier remove
print(New_dff.shape)          # after

print(dff.sample(4))          # show 4 random columns  