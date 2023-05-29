import matplotlib.pyplot as plt
import warnings
import pandas as pd



if __name__ == '__main__':
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [6, 3]
    plt.rcParams['figure.dpi'] = 300
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Identifying Credit Card Default with ML
    ## Loading data and managing data types

    df = pd.read_csv('credit_card_default.csv', index_col=0, na_values='')
    print(f'The Dataframe has {len(df)} rows and {df.shape[1]} columns.')
    print(df.head())

    x = df.copy()
    y = x.pop('default_payment_next_month')



