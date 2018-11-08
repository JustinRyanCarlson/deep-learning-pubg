import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as ker


pd.set_option('display.max_columns', None)

df = pd.read_csv('./all/train_V2.csv')
df.drop(columns=['Id', 'groupId', 'matchId', 'matchType'], inplace=True)


print('df shape: ' + str(df.shape))
# print(df.describe())
print(df.columns)
df.dropna(inplace=True)
print(df.isnull().values.any())
# print(df.dtypes.astype(str).to_dict())
targets = df['winPlacePerc']
df.drop(columns=['winPlacePerc'], inplace=True)
n_cols = df.shape[1]

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)

conf = ker.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)
ker.set_session(ker.tf.Session(config=conf))
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(df_normalized.values, targets, epochs=10)