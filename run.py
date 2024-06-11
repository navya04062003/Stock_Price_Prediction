from Core.data import Data
from Core.config import Column
from Core.model import LSTM_Trainer

def main():
    # Initialize Data object
    data = Data()
    data.read('Data/AAPL.csv')
    data.check_null_values()
    data.clean_data()
    print(Column.OPEN.value)
    data.print_head()
    data.print_description()
    scaled_data, scaler = data.normalize()
    data.visualize(Column.OPEN.value)
    data.visualize(Column.CLOSE.value)

    # Initialize LSTM Trainer
    trainer = LSTM_Trainer(data.dataframe, data.scaler)
    trainer.build_and_train_lstm()
    trainer.predict_and_plot()
    trainer.evaluate_model()

    # Save the model
    trainer.save_model('trained_model.pkl')

if __name__ == "__main__":
    main()
