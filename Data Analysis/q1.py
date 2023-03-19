import pandas as pd

train_data = pd.read_csv('drive/MyDrive/CodeCup6/travel_insurance/train.csv')
row_num, col_num = train_data.shape
avg_income = int(train_data['AnnualIncome'].mean())
traveled_num = train_data['EverTravelledAbroad'].value_counts()['Yes']
employment_info = train_data['Employment Type'].describe()
ChronicDiseases = len(train_data[train_data['ChronicDiseases'] == 1])
ChronicDiseases_TravelInsurance = len(train_data[(train_data['ChronicDiseases'] == 1) & (train_data['TravelInsurance'] == 'Yes')])

with open('output.txt', 'w') as f:
  f.write('{} {}\n'.format(row_num, col_num))
  f.write('{}\n'.format(avg_income))
  f.write('{}\n'.format(traveled_num))
  f.write('{} {:.2f}\n'.format(employment_info['top'], 100*employment_info['freq']/row_num))
  f.write('{:.2f}'.format(100*ChronicDiseases_TravelInsurance/ChronicDiseases))
