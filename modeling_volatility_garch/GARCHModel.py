import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from arch import arch_model

if __name__ == '__main__':
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [0, 4.5]
    plt.rcParams['figure.dpi'] = 300
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Volatility Forecasting
    # Explaining stock returns' volatility with ARCH Models
    # Specify the risky asset and the time horizon

    RISKY_ASSET = 'BTC-USD'
    START_DATE = '2015-01-01'
    END_DATE = '2023-05-01'

    # Download data from Yahoo Finance

    df = yf.download(RISKY_ASSET,
                     start=START_DATE,
                     end=END_DATE,
                     auto_adjust=True)
    df.index.name = None

    print(f'{df}')

    returns = 100 * df['Close'].pct_change().dropna()
    returns.name = 'asset_returns'
    returns.plot(title='Title')
    print(f'Average return: {round(returns.mean(), 2)}%')
    print(f'Returns print :{returns}')

    plt.show()

    # Specify Arch Model
    model = arch_model(returns, mean='Zero', vol='GARCH', p=1, o=0, q=1)

    # Estimate the model and print the summary

    model_fitted = model.fit(disp='off')
    print(model_fitted.summary())

    # Plot the residuals and the conditional volatility
    model_fitted.plot(annualize='D')
    plt.show()

