import pandas as pd

def format_metrics(filename):
    df = pd.read_csv(filename)
    return df.to_markdown()

if __name__ == "__main__":
    import sys
    print(format_metrics(sys.argv[1]))
