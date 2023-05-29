import yfinance as yf
import pandas as pd
import arch

if __name__ == '__main__':

    # Define the ticker symbol and the date range for the Bitcoin data
    ticker = '^GSPC'
    start_date = '2017-01-01'
    end_date = '2022-01-01'

    # Retrieve the Bitcoin price data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Extract the 'Close' prices from the data
    prices = data['Close']

    # Split the data into pre-COVID and post-COVID periods
    pre_covid_data = prices[prices.index < '2020-03-01']
    post_covid_data = prices[prices.index >= '2020-03-01']

    # Define the GARCH(1, 1) model
    model = arch.arch_model(pre_covid_data, vol='Garch', p=1, q=1)

    # Fit the model to the pre-COVID data
    result = model.fit()

    # Generate conditional variance forecasts for the pre-COVID period
    forecasts_pre_covid = result.forecast(start=pre_covid_data.shape[0])

    # Calculate the realized volatility for the pre-COVID period
    realized_volatility_pre_covid = result.conditional_volatility

    # Fit the model to the post-COVID data
    result_post_covid = model.fit(starting_values=result.params)

    # Generate conditional variance forecasts for the post-COVID period
    forecasts_post_covid = result_post_covid.forecast(start=post_covid_data.shape[0])

    # Calculate the realized volatility for the post-COVID period
    realized_volatility_post_covid = result_post_covid.conditional_volatility

    # Compare the results
    print("Pre-COVID Period:")
    print("Realized Volatility:", realized_volatility_pre_covid[-1])
    print("Conditional Volatility (GARCH(1, 1)):", forecasts_pre_covid.variance.values[-1])

    print("\nPost-COVID Period:")
    print("Realized Volatility:", realized_volatility_post_covid[-1])
    print("Conditional Volatility (GARCH(1, 1)):", forecasts_post_covid.variance.values[-1])
