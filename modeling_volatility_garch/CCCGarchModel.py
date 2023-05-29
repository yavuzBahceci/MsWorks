import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from arch import arch_model

if __name__ == '__main__':
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [0, 4.5]
    plt.rcParams['figure.dpi'] = 300
    warnings.simplefilter(action='ignore', category=FutureWarning)

    RISKY_ASSETS = ['BTC-USD', '^GSPC', 'GC=F']
    N = len(RISKY_ASSETS)
    START_DATE = '2015-01-01'
    END_DATE = '2023-05-01'

    # Download data from Yahoo Finance

    df = yf.download(RISKY_ASSETS,
                     start=START_DATE,
                     end=END_DATE,
                     auto_adjust=True)

    print(f' Downloaded {df.shape[0]} rows of data.')

    # Calculate daily returns

    returns = 100 * df['Close'].pct_change().dropna()
    returns.plot(subplots=True, title=f'Stock Returns: {START_DATE} - {END_DATE}')

    plt.show()

    # Define lists for storing objects
    coeffs = []
    cond_vol = []
    std_resids = []
    models = []

    # Estimate univariate Garch models
    for asset in returns.columns:
        # specify and fit the model
        model = arch_model(returns[asset], mean='Constant', vol='GARCH', p=1, o=0, q=1).fit(update_freq=0, disp='off')

        # Store results in the lists
        coeffs.append(model.params)
        cond_vol.append(model.conditional_volatility)
        std_resids.append(model.resid / model.conditional_volatility)
        models.append(model)

    # Store the results in Data Frames
    coeffs_df = pd.DataFrame(coeffs, index=returns.columns)
    cond_vol_df = pd.DataFrame(cond_vol).transpose() \
        .set_axis(returns.columns, axis='columns', copy=False)
    std_resids_df = pd.DataFrame(std_resids).transpose() \
        .set_axis(returns.columns, axis='columns', copy=False)

    # Calculate the constant conditional correlation matrix (R)
    R = std_resids_df.transpose() \
        .dot(std_resids_df) \
        .div(len(std_resids_df))

    # Calculate the 1-step ahead forecast of the conditional covariance matrix:

    # define objects
    diag = []
    D = np.zeros((N, N))

    # Populate the list with conditional variances
    for model in models:
        diag.append(model.forecast(horizon=1).variance.values[-1][0])

    # take the square root to obtain volatility from variance
    diag = np.sqrt(np.array(diag))
    # fill the diagonal of D with values from diag
    np.fill_diagonal(D, diag)
    # calculate the conditional covatiance matrix
    H = np.matmul(np.matmul(D, R.values), D)

    print(f'{H}')
