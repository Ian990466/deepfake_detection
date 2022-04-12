from cProfile import label
from click import style
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline

def main():
    train_df = pd.read_csv("EfficientNetCBAM_dfdc_train_roc_auc.csv")
    val_df = pd.read_csv("EfficientNetCBAM_dfdc_val_roc_auc.csv")

    train_step = train_df["Step"]/1000
    train_value = train_df["Value"]

    val_step = val_df["Step"]/1000
    val_value = val_df["Value"]

    plt.title("EfficientNet Concentrate Attention AUC")
    plt.plot (train_step, train_value, label= "Train AUC")
    plt.plot (val_step, val_value, label= "Val AUC")
    plt.xlabel("Iteration/k")
    plt.ylabel("AUC")
    
    plt.grid()
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()