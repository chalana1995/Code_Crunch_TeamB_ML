import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read data from Excel file
excel_file = "AzureUsageData.csv"
df = pd.read_csv(excel_file)

# Convert Date column to datetime format and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Resample the data to monthly frequency and sum the costs for each month
monthly_data = df['Cost'].resample('M').sum()

# Plot the monthly cost data
plt.figure(figsize=(10, 6))
plt.plot(monthly_data)
plt.title('Monthly Azure Usage Cost')
plt.xlabel('Date')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

# Plot ACF and PACF to determine the order of the SARIMA model
plot_acf(monthly_data, lags=30)
plot_pacf(monthly_data, lags=30)
plt.show()

# Define the SARIMA model with the chosen order (p, d, q) and seasonal order (P, D, Q, s)
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)  # Monthly seasonal pattern
sarima_model = SARIMAX(monthly_data, order=order, seasonal_order=seasonal_order)

# Fit the SARIMA model to the data
sarima_result = sarima_model.fit()

# Forecast future costs
forecast_steps = 12  # Forecasting for the next 12 months
forecast = sarima_result.forecast(steps=forecast_steps)

# Plot the original data and the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(monthly_data, label='Actual')
plt.plot(forecast, label='Forecast')
plt.title('Monthly Azure Usage Cost Forecast')
plt.xlabel('Date')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.show()

# Display the forecasted values
print("Forecasted Costs for the Next 12 Months:")
print(forecast)