from fastai import *
from fastai.metrics import Recall, FBeta
from fastai.tabular import *
from fastai.tabular.all import *

from fastai.tabular.core import FillMissing, Categorify, Normalize
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import tabular_learner
from fastai.tabular.model import tabular_config

from chapter_10_utils import performance_evaluation_report, custom_set_seed
import matplotlib.pyplot as plt
import warnings
import pandas as pd

if __name__ == '__main__':
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [6, 3]
    plt.rcParams['figure.dpi'] = 300
    warnings.simplefilter(action='ignore', category=FutureWarning)

    custom_set_seed(42)
    # Load the dataset from CSV file.

    df = pd.read_csv('credit_card_default.csv')
    print(df.head())

    # Identify the dependent variable(target) and numerical/categorical features:

    DEP_VAR = 'default_payment_next_month'

    num_features = list(df.select_dtypes('number').columns)
    num_features.remove(DEP_VAR)
    cat_features = list(df.select_dtypes('object').columns)

    preprocessing = [FillMissing, Categorify, Normalize]

    data = TabularDataLoaders.from_df(df,
                                      cat_names=cat_features,
                                      cont_names=num_features,
                                      procs=preprocessing,
                                      y_names=DEP_VAR,
                                      )
    print(data.show_batch(max_n=5))

    # Define the Learner object
    config = tabular_config(embed_p=0.04, ps=[0.001, 0.01])
    learn = tabular_learner(data, layers=[1000, 500],
                            config=config,
                            metrics=[Recall(),
                                     FBeta(beta=1),
                                     FBeta(beta=5)])

    # Inspect the model's architecture
    print(learn.model)
    learn.lr_find()
    learn.recorder.plot_lr_find()

    plt.show()

    # Train the neural network
    learn.fit(n_epoch=25, lr=1e-6, wd=0.2)

    # Plot the losses:
    learn.recorder.plot_loss()
    plt.tight_layout()
    plt.show()

    # Extract the predictions for the validation set
    preds_valid, _ = learn.get_preds(ds_type=DatasetType.Valid)
    pred_valid = preds_valid.argmax(dim=-1)

    # Inspect the performance ( confusion matrix ) on the validation set

