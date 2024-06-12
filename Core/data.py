import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Data:
    def __init__(self):
        self.dataframe = None
        self.scaler = None

    def read(self, file_path):
        self.dataframe = pd.read_csv(file_path)

    def check_null_values(self):
        print(self.dataframe.isnull().sum())

    def clean_data(self):
        self.dataframe.dropna(inplace=True)
        self.dataframe['date'] = pd.to_datetime(self.dataframe['date'])
        self.dataframe = self.dataframe.sort_values('date')
        self.dataframe.set_index('date', inplace=True)

    def print_head(self):
        print(self.dataframe.head())

    def print_description(self):
        print(self.dataframe.describe())

    def normalize(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataframe['normalized_close'] = self.scaler.fit_transform(self.dataframe[['close']])

    def visualize(self, column):
        plt.figure(figsize=(14, 7))
        plt.plot(self.dataframe.index, self.dataframe[column], label=column)
        plt.xlabel('Date')
        plt.ylabel(f'{column} Price')
        plt.title(f'{column} Price Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
