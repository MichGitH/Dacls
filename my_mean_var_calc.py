import csv
import yaml
import numpy as np
from pathlib import Path
import pandas as pd

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

    # Read paths from config file
    
    metrics_csv = Path(config['path']['metrics_csv'])  

def calculate_mean_variance(metrics_csv):
    data = {}

    # Read the CSV file
    with open(metrics_csv, mode='r') as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames

        # Initialize lists for each metric
        for header in headers:
            data[header] = []

        # Extract data
        for row in reader:
            for header in headers:
                data[header].append(float(row[header]))

    # Calculate mean and variance for each metric
    stats = {}
    for metric, values in data.items():
        values = np.array(values)
        mean = np.mean(values)
        variance = np.var(values)
        stats[metric] = {
            'mean': mean,
            'variance': variance
        }

    return stats


def main():

    stats = calculate_mean_variance(metrics_csv)

    # Print the calculated statistics
    for metric, stat in stats.items():
        print(f"{metric}: Mean = {stat['mean']:.4f}, Variance = {stat['variance']:.4f}")
   

    # Converti i risultati in un DataFrame pandas
    df = pd.DataFrame.from_dict(stats, orient='index')

    # Salva il DataFrame in un file Excel
    df.to_excel('media_varianza.xlsx', engine='openpyxl')

    print(f"Results have been written to media_varianza.xlsx")

if __name__ == "__main__":
    main()


