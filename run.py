from Core.data import Data
from Core.config import Column
from Core.model import LSTM_Trainer

data = Data()
data.read('AAPL.csv')
data.check_null_values()
data.clean_data()
data.print_head()
data.print_description()
data.normalize()
data.visualize(Column.OPEN.value)
data.visualize(Column.CLOSE.value)

trainer = LSTM_Trainer(data.dataframe, data.scaler)
trainer.build_and_train_lstm()
trainer.predict_and_plot()
trainer.evaluate_model()
