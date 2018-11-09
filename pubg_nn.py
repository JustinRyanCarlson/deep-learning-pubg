import pandas as pd
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

pd.set_option('display.max_columns', None)

# Load in training set and remove non-contributing columns
df = pd.read_csv('./data/train_V2.csv')
df.drop(columns=['Id', 'groupId', 'matchId', 'matchType'], inplace=True)

# Drop columns with missing data and print the shape/description
df.dropna(inplace=True)
print('df shape: ' + str(df.shape))
print('\nData Description:')
print(df.describe())

# Remove target values and scale them
targets = df['winPlacePerc']
targets = targets * 2 - 1
df.drop(columns=['winPlacePerc'], inplace=True)

# Scale model data to -1,1
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)
print("\nnp_scaled: ", np_scaled.shape, np_scaled.max(), np_scaled.min())

# Define early stopping callback function for keras model
es_cb = EarlyStopping(monitor='val_loss', min_delta=0, mode='auto', patience=2)

# Construct and fit model
model = Sequential()
model.add(Dense(150, activation='relu', input_shape=(df.shape[1],)))
model.add(Dense(75, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model.summary()
model.fit(np_scaled, targets, epochs=50, verbose=1, batch_size=1000, validation_split=0.3, callbacks=[es_cb])

# Read in test data and scale to match the training data
df_test = pd.read_csv('./data/test_V2.csv')
df_test.drop(columns=['Id', 'groupId', 'matchId', 'matchType'], inplace=True)
np_scaled_test = min_max_scaler.fit_transform(df_test)
print("np_scaled_test", np_scaled_test.shape, np_scaled_test.max(), np_scaled_test.min())

# Predict test data target values and rescale to 0-1 range
pred = model.predict(np_scaled_test)
pred = pred.reshape(-1)
pred = (pred + 1) / 2

# Reformat winPlacePerc
print("Fix winPlacePerc")
for i in range(len(df_test)):
    winPlacePerc = pred[i]
    maxPlace = int(df_test.iloc[i]['maxPlace'])
    if maxPlace == 0:
        winPlacePerc = 0.0
    elif maxPlace == 1:
        winPlacePerc = 1.0
    else:
        gap = 1.0 / (maxPlace - 1)
        winPlacePerc = round(winPlacePerc / gap) * gap
    
    if winPlacePerc < 0: winPlacePerc = 0.0
    if winPlacePerc > 1: winPlacePerc = 1.0    
    pred[i] = abs(winPlacePerc)

print('Saving results')
df_test = pd.read_csv('./data/test_V2.csv')
df_test['winPlacePerc'] = pred
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)