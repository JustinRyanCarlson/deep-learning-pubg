import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as ker

pd.set_option('display.max_columns', None)

df = pd.read_csv('../input/train_V2.csv')
df.drop(columns=['Id', 'groupId', 'matchId', 'matchType'], inplace=True)

print('df shape: ' + str(df.shape))
# print(df.describe())
print(df.columns)
df.dropna(inplace=True)
print(df.isnull().values.any())
# print(df.dtypes.astype(str).to_dict())
targets = df['winPlacePerc']
targets = targets * 2 - 1
print(targets)
df.drop(columns=['winPlacePerc'], inplace=True)
n_cols = df.shape[1]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)
print("np_scaled", np_scaled.shape, np_scaled.max(), np_scaled.min())
# exit(0)

conf = ker.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)
ker.set_session(ker.tf.Session(config=conf))

es_cb = EarlyStopping(monitor='val_loss', min_delta=0, mode='auto', patience=2)

model = Sequential()
model.add(Dense(300, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
model.summary()
model.fit(np_scaled, targets, epochs=50, verbose=1, batch_size=1000, validation_split=0.3, callbacks=[es_cb])



df_test = pd.read_csv('../input/test_V2.csv')
df_test.drop(columns=['Id', 'groupId', 'matchId', 'matchType'], inplace=True)
np_scaled_test = min_max_scaler.fit_transform(df_test)
print("np_scaled_test", np_scaled_test.shape, np_scaled_test.max(), np_scaled_test.min())

pred = model.predict(np_scaled_test)
pred = pred.reshape(-1)
pred = (pred + 1) / 2

print("fix winPlacePerc")
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
    pred[i] = winPlacePerc

print('saving')
df_test = pd.read_csv('../input/test_V2.csv')
df_test['winPlacePerc'] = pred
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)