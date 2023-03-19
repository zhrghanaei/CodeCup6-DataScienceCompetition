from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from parsivar import Normalizer, Tokenizer, FindStems
import pandas as pd

train_df = pd.read_csv('comments/train.csv')
test_df = pd.read_csv('comments/test.csv')

my_normalizer = Normalizer(pinglish_conversion_needed = True)
my_stemmer = FindStems()
punctuations = {ord(c): '' for c in '''.!()-[]{};:'"\,<>./?؟@#$%^&*_~'''}
numbers = {ord(c): '' for c in '1234567890'}
space = {'\u200c': ' '}
with open('stopwords.txt') as f:
    stopwords = [' '+w.strip()+' ' for w in f.readlines()]

def clean_text(row):
  text = my_normalizer.normalize(row['comment'])
  text = text.translate(punctuations)
  text = text.translate(numbers)
  text = text.translate(space)
  # text = ' '.join(my_stemmer.convert_to_stem(w).split('&')[0] for w in text.split())
  text = ' ' + text + ' '
  for w in stopwords:
    text = text.replace(w, ' ')
  # text = text.split()
  row['comment'] = text
  return row

train_df = train_df.apply(clean_text, axis = 1)
test_df = test_df.apply(clean_text, axis = 1)

vectorizer = TfidfVectorizer(lowercase = False, ngram_range = (1,2))
train_vec = vectorizer.fit_transform(train_df.comment.to_list())
test_vec = vectorizer.transform(test_df.comment.to_list())

knn_model = KMeans(n_clusters=2, init='k-means++', max_iter=2000, n_init=1)
knn_model.fit(train_vec)

labels = knn_model.predict(train_vec)

model = XGBClassifier(n_estimators = 100, random_state=101)
model.fit(train_vec, labels)

predictions = model.predict_proba(test_vec)
out_df = pd.DataFrame(predictions[:,1], columns = ['prediction'])
out_df.to_csv('output_comment_XGBClassifier.csv', index=False)
