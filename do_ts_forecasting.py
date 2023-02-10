import argparse
import pandas as pd
import numpy as np

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-m',
      '--model',
      type=str,
      default='ses',
      help=('Model names: ses | holts | holts_damped | auto_arima | '
            'thetaf | tbats'))
  parser.add_argument('--train', type=pd.Series, default=None, help='Time-series training data')
  parser.add_argument('-y', '--y_data', type=pd.Series, default=None, help='Time-series raw data')
  parser.add_argument('--test', type=pd.Series, default=None, help='Time-series test data')
  parser.add_argument('-f', '--forecast_steps', type=int, default=5, help='Forecast steps into the future')
  
  args = parser.parse_args()

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

  if test is None:
    rmse, mape, mae = None, None, None
  else:
    rmse = np.sqrt(mean_squared_error(forecast, test)).round(2)
    mape = np.round(np.mean(np.abs(forecast - args.test)/np.abs(args.test))*100,2)
    mae = mean_absolute_error(args.test, forecast)
  

  output = {
      "model_name":args.model,
      "y_pred" : forecast,
      "rmse": rmse if rmse is not None else "None",
      "mape": mape if mape is not None else "None",
      "mae": mae if mae is not None else "None",
      "plot": forecast_plot(args.y_data, args.test, forecast)}

  return output

if __name__ == '__main__':
  main()
