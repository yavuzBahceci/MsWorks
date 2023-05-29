import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (Dataset, TensorDataset, DataLoader, Subset)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 8.Define the network's architecture
class MLP(nn.Module):

    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x


# 4. Define a function for transforming time series into a dataset for the MLP:
def create_input_data(series, n_lags=1):
    """
    Function for transforming time series into input acceptable by a multilayer perceptron
    :param series: np.array
    The time series to be transformed
    :param n_lags: Int
    The number of lagged observations to consider as features
    :return:
    X: np.array
    array of features
    y: np.array
    array of Target
    """
    X, y = [], []

    for step in range(len(series) - n_lags):
        end_step = step + n_lags
        X.append(series[step:end_step])
        y.append(series[end_step])

    return np.array(X), np.array(y)


if __name__ == '__main__':
    print(torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Define Parameters
    TICKER = 'ANF'
    START_DATE = '2010-01-02'
    END_DATE = '2019-12-31'
    N_LAGS = 3

    # neural network =
    VALID_SIZE = 12
    BATCH_SIZE = 5
    N_EPOCHS = 1000

    # 3. Download and prepare the data

    df = yf.download(TICKER.strip(), start=START_DATE, end=END_DATE, progress=False)

    print(df)
    df = df.resample('M').last()
    prices = df['Adj Close']

    fig, ax = plt.subplots()
    ax.plot(df.index, prices)
    ax.set(title=f"{TICKER}'s Stock price", xlabel='Time', ylabel='Price ($)')

    plt.show()

    # 5.Transform the considered time series into input for the MLP:
    X, y = create_input_data(prices, N_LAGS)
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().unsqueeze(dim=1)

    # 6. Create training and validation sets:
    valid_ind = len(X) - VALID_SIZE
    dataset = TensorDataset(X_tensor, y_tensor)

    train_dataset = Subset(dataset, list(range(valid_ind)))
    valid_dataset = Subset(dataset, list(range(valid_ind, len(X))))

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

    # Inspect the observations from the first batch
    print(next(iter(train_loader))[0])
    print(next(iter(train_loader))[1])

    # Check the size of datasets:
    print(F'size of datasets - training: {len(train_loader.dataset)} | validation: {len(valid_loader.dataset)}')

    # Use a naive forecast as a benchmark and evaluate the performance:
    naive_pred = prices[len(prices) - VALID_SIZE - 1:-1]
    y_valid = prices[len(prices) - VALID_SIZE:]

    naive_mse = mean_squared_error(y_valid, naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    print(f'Naive Forecast - MSE: {naive_mse:.2f}, RMSE: {naive_rmse:.2f}')

    # Testing Linear Regression
    X_train = X[:valid_ind, ]
    y_train = y[:valid_ind]

    X_valid = X[valid_ind:, ]
    y_valid = y[valid_ind:]

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_valid)
    lr_mse = mean_squared_error(y_valid, y_pred)
    ln_rmse = np.sqrt(lr_mse)
    print(F"Linear Regression's forecast - MSE {lr_mse:.2f}, RMSE: {ln_rmse:.2f}")
    print(f"Linear regression's coefficients: {lin_reg.coef_}")

    fig, ax = plt.subplots()

    ax.plot(y_valid, color='blue', label='Actual')
    ax.plot(y_pred, color='red', label='Prediction')

    ax.set(title="Linear Regression's forecast",
           xlabel='Time',
           ylabel="Price ($)")
    ax.legend()

    plt.show()

    # 9. Instantiate the model, the loss function and the optimizer:

    # set seed for reproducibility
    torch.manual_seed(42)

    model = MLP(N_LAGS).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model)

    # 10. Train the network
    PRINT_EVERY = 50
    train_losses, valid_losses = [], []

    for epoch in range(N_EPOCHS):
        running_loss_train = 0
        running_loss_valid = 0

        model.train()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model(x_batch)
            loss = loss_fn(y_batch, y_hat)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * x_batch.size(0)

        epoch_loss_train = running_loss_train / len(train_loader.dataset)
        train_losses.append(epoch_loss_train)

        with torch.no_grad():
            model.eval()

            for x_val, y_val in valid_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_hat = model(x_val)
                loss = loss_fn(y_val, y_hat)
                running_loss_valid += loss.item() * x_val.size(0)

            epoch_loss_valid = running_loss_valid / len(valid_loader.dataset)

            if epoch > 0 and epoch_loss_valid < min(valid_losses):
                best_epoch = epoch
                torch.save(model.state_dict(), 'mlp_checkpoint.pth')

            valid_losses.append(epoch_loss_valid)

        if epoch % PRINT_EVERY == 0:
            print(f"<{epoch}> - Train. loss: {epoch_loss_train:.2f} \t Valid. loss: {epoch_loss_valid:.2f}")

    print(f"Lowest loss recorded in epoch: {best_epoch}")

    # 11. Plot the losses over epochs:

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots()

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')

    ax.set(title='Loss over epochs',
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 12. Load the best model (with the lowest validation loss):

    state_dict = torch.load('mlp_checkpoint.pth')
    model.load_state_dict(state_dict)

    # 13. Obtain the predictions
    y_pred, y_valid = [], []

    with torch.no_grad():
        model.eval()
        for x_val, y_val in valid_loader:
            x_val = x_val.to(device)
            y_pred.append(model(x_val))
            y_valid.append(y_val)

    y_pred = torch.cat(y_pred).numpy().flatten()
    y_valid = torch.cat(y_valid).numpy().flatten()

    # 14. Evaluate the predictions:

    mlp_mse = mean_squared_error(y_valid, y_pred)
    mlp_rmse = np.sqrt(mlp_mse)
    print(f"MLP's forecast - MSE : {mlp_mse:.2f}, RMSE: {mlp_rmse:.2f}")

    fig, ax = plt.subplots()

    ax.plot(y_valid, color='blue', label='True')
    ax.plot(y_pred, color='red', label='Prediction')

    ax.set(title="Multilayer Perceptron's Forecasts",
           xlabel='Time',
           ylabel='Price ($)')
    ax.legend()

    plt.tight_layout()
    plt.show()
