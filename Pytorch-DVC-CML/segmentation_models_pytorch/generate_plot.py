import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_metrics(metrics_file, title):
    df = pd.read_csv(metrics_file, sep="\t")
    
    df.plot()
    plt.title(title)
    plt.savefig(f'{title}.png')

if __name__ == "__main__":
    plot_metrics(sys.argv[1], sys.argv[2])

