import pandas as pd

cab_train = pd.read_csv('cab/train.csv')
cab_test = pd.read_csv('cab/test.csv')

cab_train['weekday'] = pd.to_datetime(cab_train['time']).dt.weekday
cab_train['hms'] = pd.to_datetime(cab_train['time']).dt.time

cab_test['weekday'] = pd.to_datetime(cab_test['time']).dt.weekday
cab_test['hms'] = pd.to_datetime(cab_test['time']).dt.time

time_preds_med = dict(cab_train.groupby(['weekday', 'hms'])['y'].median())
time_preds_mean = dict(cab_train.groupby(['weekday', 'hms'])['y'].mean())

cab_test['prediction'] = cab_test.apply(lambda row: 1.2 * ((time_preds_mean[(row['weekday'], row['hms'])] + time_preds_med[(row['weekday'], row['hms'])]) / 2), axis = 1)

cab_test.to_csv('output_mean_med.csv', columns = ['prediction'], index = False)
