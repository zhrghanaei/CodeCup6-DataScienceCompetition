#@title train data preprocessing
import pandas as pd
import numpy as np

hotels_train = pd.read_csv('hotels/train.csv')

hotels_train.drop('user', axis=1, inplace=True)
hotels_train.drop('destination', axis=1, inplace=True)

channels = hotels_train['channel'].unique()
# destinations = hotels_train['destination'].unique()
hotel_categories = hotels_train['hotel_category'].unique()

channel_dict = dict(zip(channels, [i/(len(channels)-1) for i in range(len(channels))]))
# destination_dict = dict(zip(destinations, [i/(len(destinations)-1) for i in range(len(destinations))]))
hotel_category_dict = dict(zip(hotel_categories, [i/(len(hotel_categories)-1) for i in range(len(hotel_categories))]))

hotels_train['search_date'] = (pd.to_datetime(hotels_train['search_date']).dt.dayofyear) / 365
hotels_train['checkIn_date'] = (pd.to_datetime(hotels_train['checkIn_date']).dt.dayofyear).fillna(0) / 365
hotels_train['checkOut_date'] = (pd.to_datetime(hotels_train['checkOut_date']).dt.dayofyear).fillna(0) / 365

# resample data to make dataset balanced
pos = list(hotels_train.index[hotels_train['is_booking'] == True])
idx = pos + list(np.random.choice(hotels_train.index[hotels_train['is_booking'] == False].tolist(), size = len(pos), replace = False))
sampled_df = hotels_train.iloc[idx]

# preprocessing
sampled_df['time_to_trip'] = sampled_df.apply(lambda row: (row['checkIn_date'] - row['search_date']), axis = 1)
sampled_df['trip_len'] = sampled_df.apply(lambda row: (row['checkOut_date'] - row['checkIn_date']), axis = 1)
sampled_df['channel'].replace(channel_dict, inplace= True)
# sampled_df['destination'].replace(destination_dict, inplace= True)
sampled_df['hotel_category'].replace(hotel_category_dict, inplace= True)
sampled_df.replace({True : 1, False : 0}, inplace= True)

# change dtypes to reduce memory usage
sampled_df.search_date = sampled_df.search_date.astype('float32')
sampled_df.channel = sampled_df.channel.astype('float32')
sampled_df.is_mobile = sampled_df.is_mobile.astype('float32')
sampled_df.is_package = sampled_df.is_package.astype('float32')
# sampled_df.destination = sampled_df.destination.astype('category')
sampled_df.checkIn_date = sampled_df.checkIn_date.astype('float32')
sampled_df.checkOut_date = sampled_df.checkOut_date.astype('float32')
sampled_df.n_adults = sampled_df.n_adults.astype('float32')
sampled_df.n_children = sampled_df.n_children.astype('float32')
sampled_df.n_rooms = sampled_df.n_rooms.astype('float32')
sampled_df.hotel_category = sampled_df.hotel_category.astype('float32')
sampled_df.is_booking = sampled_df.is_booking.astype('float32')
sampled_df.time_to_trip = sampled_df.time_to_trip.astype('float32')
sampled_df.trip_len = sampled_df.trip_len.astype('float32')

del hotels_train

#@title MLP Model
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

BATCH_SIZE = 1024

val_dataframe = sampled_df.sample(frac=0.2)
train_dataframe = sampled_df.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop('is_booking')
    ds = tf.data.Dataset.from_tensor_slices((dataframe, labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

all_inputs = keras.Input(shape=(12,), name="input")
x = layers.Dense(150, activation="relu")(all_inputs)
x = layers.Dense(75, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(30, activation="relu")(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation="sigmoid")(x)
mlp_model = keras.Model(all_inputs, output)
mlp_model.compile(Adam(lr=0.1), "binary_crossentropy", metrics=[tf.keras.metrics.AUC()])

checkpoint_filepath = '/model/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

mlp_model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=[model_checkpoint_callback])
# print(mlp_model.summary())

#@title Test data preprocessing
hotels_test = pd.read_csv('hotels/test.csv')

hotels_test.drop('user', axis=1, inplace=True)
hotels_test.drop('destination', axis=1, inplace=True)

hotels_test['search_date'] = (pd.to_datetime(hotels_test['search_date']).dt.dayofyear) / 365
hotels_test['checkIn_date'] = (pd.to_datetime(hotels_test['checkIn_date']).dt.dayofyear).fillna(0) / 365
hotels_test['checkOut_date'] = (pd.to_datetime(hotels_test['checkOut_date']).dt.dayofyear).fillna(0) / 365

# preprocessing
hotels_test['time_to_trip'] = hotels_test.apply(lambda row: (row['checkIn_date'] - row['search_date']), axis = 1)
hotels_test['trip_len'] = hotels_test.apply(lambda row: (row['checkOut_date'] - row['checkIn_date']), axis = 1)
hotels_test['channel'].replace(channel_dict, inplace= True)
# hotels_test['destination'].replace(destination_dict, inplace= True)
hotels_test['hotel_category'].replace(hotel_category_dict, inplace= True)
hotels_test.replace({True : 1, False : 0}, inplace= True)

# change dtypes to reduce memory usage
hotels_test.search_date = hotels_test.search_date.astype('float32')
hotels_test.channel = hotels_test.channel.astype('float32')
hotels_test.is_mobile = hotels_test.is_mobile.astype('float32')
hotels_test.is_package = hotels_test.is_package.astype('float32')
# hotels_test.destination = hotels_test.destination.astype('category')
hotels_test.checkIn_date = hotels_test.checkIn_date.astype('float32')
hotels_test.checkOut_date = hotels_test.checkOut_date.astype('float32')
hotels_test.n_adults = hotels_test.n_adults.astype('float32')
hotels_test.n_children = hotels_test.n_children.astype('float32')
hotels_test.n_rooms = hotels_test.n_rooms.astype('float32')
hotels_test.hotel_category = hotels_test.hotel_category.astype('float32')
hotels_test.time_to_trip = hotels_test.time_to_trip.astype('float32')
hotels_test.trip_len = hotels_test.trip_len.astype('float32')

#@title create output
mlp_model.load_weights(checkpoint_filepath)
predictions = mlp_model.predict(hotels_test)
out_df = pd.DataFrame(predictions, columns = ['prediction'])
out_df.to_csv('output_hotel_final.csv', index=False)
