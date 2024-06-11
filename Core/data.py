import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Data:
    def _init_(self):
        self.dataframe = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def read(self, file_path):
        self.dataframe = pd.read_csv(file_path)
        self.dataframe[Column.DATE.value] = pd.to_datetime(self.dataframe[Column.DATE.value])
        self.dataframe.set_index(Column.DATE.value, inplace=True)

    def check_null_values(self):
        print(self.dataframe.isnull().sum())

    def clean_data(self):
        self.dataframe.dropna(inplace=True)

    def normalize(self):
        self.dataframe[Column.CLOSE.value] = self.scaler.fit_transform(self.dataframe[[Column.CLOSE.value]])
        return self.dataframe, self.scaler

    def print_head(self):
        print(self.dataframe.head())

    def print_description(self):
        print(self.dataframe.describe())

    def visualize(self, column):
        plt.figure(figsize=(14, 7))
        plt.plot(self.dataframe.index, self.dataframe[column])
        plt.xlabel('Date')
        plt.ylabel(f'{column.capitalize()} Price (USD)')
        plt.title(f'{column.capitalize()} Price Over Time')
        plt.grid(True)
        plt.show()
