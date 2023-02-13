import argparse
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.forecasting.theta import ThetaModel
from pmdarima.arima import auto_arima
from tbats import BATS, TBATS
from scipy.stats import boxcox
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import base64
import io

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-m',
      '--model',
      type=str,
      default='ses',
      help=('Model names: ses | holts | holts_damped | auto_arima | '
            'thetaf | tbats'))
  parser.add_argument('-d','--data', type=str, default=None, help='Time-series training data')
  parser.add_argument('-f', '--forecast_steps', type=int, default=5, help='Forecast steps into the future')
  
  args = parser.parse_args()
  
  print(exec(args.data))
  
  print(data["train"])
  print(data["y_data"])
  train = pd.Series(data["train"])
  y_data = pd.Series(data["y_data"])
  
  list_of_models = {
      "ses" : "SimpleExpSmoothing(train).fit()",
      "holts" : "Holt(train).fit()",
      "holts_damped" : "ExponentialSmoothing(train, trend = 'add', damped_trend = True).fit()",
      "auto_arima" : "auto_arima(train, stepwise = False, maxiter = 100)",
      "thetaf" : "ThetaModel(train).fit()",
      "tbats" : "TBATS().fit(train)"
  }

  model = eval(list_of_models[args.model])

  if args.model == "auto_arima":
    forecast = model.predict(n_periods = args.forecast_steps)
  else:
    forecast = model.forecast(args.forecast_steps)

  if not isinstance(forecast, pd.Series):
    forecast = pd.Series(forecast)

  if data["test"][0] == 'None':
    test = None
    rmse, mape, mae = None, None, None
  else:
    test = pd.Series(data["test"][0])
    rmse = np.sqrt(mean_squared_error(forecast, test)).round(2)
    mape = np.round(np.mean(np.abs(forecast - test)/np.abs(test))*100,2)
    mae = mean_absolute_error(test, forecast)
  

  output = {
      "model_name":args.model,
      "y_pred" : forecast,
      "rmse": rmse if rmse is not None else "None",
      "mape": mape if mape is not None else "None",
      "mae": mae if mae is not None else "None",
      "plot": forecast_plot(y_data, test, forecast)}

  return output

if __name__ == '__main__':
  main()

  
def forecast_plot(y_data, test, forecast):
  plt.figure(figsize=(16, 8), dpi=150)
  
  if test is not None:
    test.plot(label='test')

  y_data.plot(label='y_data', color='orange')
  forecast.plot(label='forecast')
  plt.title('Forecast Plot')
  plt.xlabel('Weeks')
  plt.legend()

  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  buf.seek(0)
  plt.close()

  return base64.b64encode(buf.read())
