import pandas as pd
from surprise import Dataset, Reader, SVD

train_df = pd.read_csv('data/train.csv', dtype={'itemId': object})
test_df = pd.read_csv('data/test.csv', dtype={'itemId': object})

train_df.drop('date', axis = 1, inplace = True)
train_df['rating'] = train_df['rating'].astype('uint8')
test_df.drop('date', axis = 1, inplace = True)

reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(train_df[["userId", "itemId", "rating"]], reader)
trainset = train_data.build_full_trainset()
algo = SVD(verbose = True)
algo.fit(trainset)

output = pd.DataFrame()
output['prediction'] = test_df.apply(lambda row: algo.predict(row['userId'], row['itemId']).est, axis = 1)
output['prediction'].to_csv('output_recommender.csv', index = False)
