import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, IntegerLookup

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode='binary')

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

BATCH_SIZE = 32

train_df = pd.read_csv('drive/MyDrive/CodeCup6/travel_insurance/train.csv')
test_df = pd.read_csv('drive/MyDrive/CodeCup6/travel_insurance/test.csv')

val_dataframe = train_df.sample(frac=0.2)
train_dataframe = train_df.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    dataframe['TravelInsurance'].replace(['Yes', 'No'], [1,0], inplace=True)
    labels = dataframe.pop('TravelInsurance')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

# Categorical feature encoded as string
EmploymentType = keras.Input(shape=(1,), name="Employment Type", dtype="string")
GraduateOrNot = keras.Input(shape=(1,), name="GraduateOrNot", dtype="string")
FrequentFlyer = keras.Input(shape=(1,), name="FrequentFlyer", dtype="string")
EverTravelledAbroad = keras.Input(shape=(1,), name="EverTravelledAbroad", dtype="string")
ChronicDiseases = keras.Input(shape=(1,), name="ChronicDiseases")

# Numerical features
Age = keras.Input(shape=(1,), name="Age")
AnnualIncome	 = keras.Input(shape=(1,), name="AnnualIncome")
FamilyMembers = keras.Input(shape=(1,), name="FamilyMembers")


all_inputs = [
    EmploymentType,
    GraduateOrNot,
    FrequentFlyer,
    EverTravelledAbroad,
    ChronicDiseases,
    Age,
    AnnualIncome,
    FamilyMembers
]

# String categorical features
EmploymentType_encoded = encode_categorical_feature(EmploymentType, "Employment Type", train_ds, True)
GraduateOrNot_encoded = encode_categorical_feature(GraduateOrNot, "GraduateOrNot", train_ds, True)
FrequentFlyer_encoded = encode_categorical_feature(FrequentFlyer, "FrequentFlyer", train_ds, True)
EverTravelledAbroad_encoded = encode_categorical_feature(EverTravelledAbroad, "EverTravelledAbroad", train_ds, True)
ChronicDiseases_encoded = encode_categorical_feature(ChronicDiseases, "ChronicDiseases", train_ds, False)

# Numerical features
Age_encoded = encode_numerical_feature(Age, "Age", train_ds)
AnnualIncome_encoded = encode_numerical_feature(AnnualIncome, "AnnualIncome", train_ds)
FamilyMembers_encoded = encode_numerical_feature(FamilyMembers, "FamilyMembers", train_ds)


all_features = layers.concatenate(
    [
      EmploymentType_encoded,
      GraduateOrNot_encoded,
      FrequentFlyer_encoded,
      EverTravelledAbroad_encoded,
      ChronicDiseases_encoded,
      Age_encoded,
      AnnualIncome_encoded,
      FamilyMembers_encoded
    ]
)

x = layers.Dense(250, activation="relu")(all_features)
x = layers.Dense(150, activation="relu")(x)
x = layers.Dense(75, activation="relu")(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

checkpoint_filepath = '/model/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[model_checkpoint_callback])
# print(model.summary())

model.load_weights(checkpoint_filepath)
predictions = model.predict(dict(test_df))
out_df['Customer Id'] = test_df['Customer Id']
out_df['prediction'] = pd.DataFrame(predictions)
out_df.to_csv('output.csv', index=False)
