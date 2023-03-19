import pandas as pd

supermarket_data = pd.read_csv('drive/MyDrive/CodeCup6/supermarket.csv')
supermarket_data['Date'] = pd.to_datetime(supermarket_data['Date'], format='%Y-%m-%d')
supermarket_data['Year'] = supermarket_data['Date'].dt.year
supermarket_data['WeekDay'] = supermarket_data['Date'].dt.day_name()

unique_products = supermarket_data['Product'].nunique()
avegare_per_day = supermarket_data['Date'].value_counts().mean()
min_freq_products = ','.join(supermarket_data['Product'].value_counts(ascending = True)[:4].index)
max_customer = ','.join(supermarket_data[supermarket_data['Year'] == 2020]['Customer Id'].value_counts()[:5].index)
most_freq_day = supermarket_data['WeekDay'].value_counts().index[0]

with open('output.txt', 'w') as f:
  f.write('{}\n'.format(unique_products))
  f.write('{:.2f}\n'.format(avegare_per_day))
  f.write('{}\n'.format(min_freq_products))
  f.write('{}\n'.format(max_customer))
  f.write('{}\n'.format(most_freq_day))
  f.write('{}\n'.format(-1))
  f.write('{}\n'.format(-1))
  f.write('{}'.format(-1))            
