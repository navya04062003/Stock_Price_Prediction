import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import pickle

class LSTM_Trainer:
    def _init_(self, dataframe, scaler):
        self.dataframe = dataframe
        self.scaler = scaler
        self.model = None
        self.x_train, self.y_train, self.x_test, self.y_test_actual = self.prepare_data()

    def prepare_data(self, time_step=60):
        data = self.dataframe[Column.CLOSE.value].values
        x, y = [], []
        for i in range(time_step, len(data)):
            x.append(data[i-time_step:i])
            y.append(data[i])
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        train_size = int(len(x) * 0.8)
        x_train, y_train = x[:train_size], y[:train_size]
        x_test, y_test_actual = x[train_size:], y[train_size:]
        
        return x_train, y_train, x_test, y_test_actual

    def build_and_train_lstm(self, epochs=100, batch_size=64):
        self.model = Sequential()
        self.model.add(LSTM(100, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(100, return_sequences=False))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

    def predict_and_plot(self):
        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(self.y_test_actual.reshape(-1, 1))

        plt.figure(figsize=(14, 7))
        plt.plot(self.dataframe.index[-len(y_test_actual):], y_test_actual, label='Actual Prices')
        plt.plot(self.dataframe.index[-len(predictions):], predictions, label='Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title('Actual vs Predicted Prices')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(self.y_test_actual.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mse = mean_squared_error(y_test_actual, predictions)

        print(f'RMSE: {rmse}')
        print(f'MSE: {mse}')

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)
