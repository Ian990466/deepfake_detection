from cProfile import label
from click import style
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline

def main():
    train_df = pd.read_csv("dfcd-train-loss.csv")
    val_df = pd.read_csv("dfdc-val-loss.csv")

    train_step = train_df["Step"]/1000
    train_value = train_df["Value"]

    val_step = val_df["Step"]/1000
    val_value = val_df["Value"]

    plt.figure(dpi=160)
    plt.title("EfficientNetFIA Loss")
    plt.plot (train_step, train_value, label= "Train Loss")
    plt.plot (val_step, val_value, label= "Val Loss")
    plt.xlabel("Iteration/k")
    plt.ylabel("LogLoss Value")
    
    plt.grid()
    plt.legend()
    
    plt.savefig('.png') 

    # plt.show()

if __name__ == "__main__":
    main()