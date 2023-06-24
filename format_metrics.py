import pandas as pd

def format_metrics(filename):
    df = pd.read_csv(filename)
    # adjust column width
    #pd.set_option('display.max_colwidth', 100)  # adjust as needed
    return df.to_html()

if __name__ == "__main__":
    import sys
    print(format_metrics(sys.argv[1]))
