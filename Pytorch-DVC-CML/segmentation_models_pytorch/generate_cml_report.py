import json
import pandas as pd
import sys

def generate_cml_report(metrics_file):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Create a DataFrame from the metrics
    df = pd.DataFrame([metrics])

    # Convert the DataFrame to Markdown
    report = df.to_markdown()

    return report

if __name__ == "__main__":
    print(generate_cml_report(sys.argv[1]))

