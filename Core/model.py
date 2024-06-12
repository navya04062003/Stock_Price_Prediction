import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error

class LSTM_Trainer:
    def __init__(self, dataframe, scaler):
        self.dataframe = dataframe
        self.scaler = scaler
        self.model = None
        self.history = None

    def prepare_data(self, data, time_step=60):
        x, y = [], []
        for i in range(time_step, len(data)):
            x.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        return x, y

    def build_and_train_lstm(self, epochs=40, batch_size=64, time_step=60):
        # Prepare training and validation data
        scaled_data = self.dataframe['normalized_close'].values.reshape(-1, 1)
        train_size = int(len(scaled_data) * 0.7)
        val_size = int(len(scaled_data) * 0.15)
        train_data = scaled_data[:train_size]
        val_data = scaled_data[train_size:train_size + val_size]
        x_train, y_train = self.prepare_data(train_data, time_step)
        x_val, y_val = self.prepare_data(val_data, time_step)

        # Build LSTM model
        self.model = Sequential()
        self.model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(LSTM(units=60, return_sequences=False))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)

    def plot_training_history(self):
        # Plot training & validation loss values
        plt.figure(figsize=(14, 7))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_and_plot(self, time_step=60):
        scaled_data = self.dataframe['normalized_close'].values.reshape(-1, 1)
        test_data = scaled_data[int(len(scaled_data) * 0.85) - time_step:]
        x_test, y_test_actual = self.prepare_data(test_data, time_step)

        # Make predictions on test data
        test_predictions = self.model.predict(x_test)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        y_test_actual = self.scaler.inverse_transform(y_test_actual.reshape(-1, 1))

        # Plot training predictions vs actual
       plt.figure(figsize=(14, 7))
       plt.plot(self.dataframe.index[60:60+len(y_train_actual)], y_train_actual, label='Actual Training Observations', color='blue')
       plt.plot(self.dataframe.index[60:60+len(train_predictions)], train_predictions, label='Training Predictions', color='orange')
       plt.xlabel('Date')
       plt.ylabel('Adjusted Close Price (Scaled)')
       plt.title('Training Predictions vs Actual Training Observations')
       plt.legend()
       plt.grid(True)
       plt.show()


    def evaluate_model(self, time_step=60):
        scaled_data = self.dataframe['normalized_close'].values.reshape(-1, 1)
        train_data = scaled_data[:int(len(scaled_data) * 0.7)]
        x_train, y_train = self.prepare_data(train_data, time_step)
        train_predictions = self.model.predict(x_train)
        train_predictions = self.scaler.inverse_transform(train_predictions)
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))

        test_data = scaled_data[int(len(scaled_data) * 0.85) - time_step:]
        x_test, y_test_actual = self.prepare_data(test_data, time_step)
        test_predictions = self.model.predict(x_test)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        y_test_actual = self.scaler.inverse_transform(y_test_actual.reshape(-1, 1))

        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
        train_mse = mean_squared_error(y_train_actual, train_predictions)
        print(f'Training RMSE: {train_rmse}')
        print(f'Training MSE: {train_mse}')

        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
        test_mse = mean_squared_error(y_test_actual, test_predictions)
        print(f'Test RMSE: {test_rmse}')
        print(f'Test MSE: {test_mse}')
