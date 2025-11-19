import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

train = pd.read_csv("Google_Stock_Price_Train.csv")
test = pd.read_csv("Google_Stock_Price_Test.csv")

train_data = train[['Close']].values
test_data = test[['Close']].values

sc = MinMaxScaler(feature_range=(0,1))
train_scaled = sc.fit_transform(train_data)

X_train = []
y_train = []
for i in range(60, len(train_scaled)):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=32)

dataset_total = np.concatenate((train_data, test_data), axis=0)
inputs = dataset_total[len(dataset_total)-len(test_data)-60:]
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

test['Predicted'] = predicted_price

last_day_actual = test['Close'].iloc[-1]
next_day_pred = predicted_price[-1][0]

if next_day_pred > last_day_actual:
    print("Next Day Trend: INCREASE")
else:
    print("Next Day Trend: DECREASE")

print("Last Actual Close:", last_day_actual)
print("Next Day Predicted Close:", next_day_pred)
