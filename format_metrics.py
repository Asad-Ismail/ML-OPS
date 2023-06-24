import json
import pandas as pd
import sys

def format_metrics(filename):
    with open(filename, "r") as read_file:
        data = json.load(read_file)

    df = pd.json_normalize(data)
    df.index = ["Metrics"]
    pd.set_option('display.max_colwidth', -1)
    return df.to_markdown()

if __name__ == "__main__":
    print(format_metrics(sys.argv[1]))

