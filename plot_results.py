import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import os


def plot_box(df, x, y, title, ylabel, save_dir, filename):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.xlabel(x.replace('_', ' ').title())
    plt.ylabel(ylabel)
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {save_path}")
    plt.close() 


def main():
    # DONE: load the "results.csv" file from the "mia-result" LATEST directory
    # DONE: read the data into a list by using pandas
    # DONE: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot


    # Automatically get the latest results.csv file
    base_dir = os.path.join(os.path.dirname(__file__), "mia-result")
    csv_paths = sorted(glob.glob(os.path.join(base_dir, "*", "results.csv")), reverse=True)

    if not csv_paths:
        raise FileNotFoundError(f"No results.csv files found in {base_dir}")
        
    latest_csv = csv_paths[0]
    print(f"Loading results from: {latest_csv}")

    # Load CSV
    df = pd.read_csv(latest_csv,  sep=';')

    print(df)                 # print the first 5 rows, and the last 5 rows
    # print(df.to_string())     # print all the rows

    save_dir = os.path.dirname(latest_csv)
    plot_box(df, 'LABEL', 'DICE', 'Distribution of DICE Scores by Anatomical Structure (higher is better)', 'DICE Score', save_dir, 'dice_scores_boxplot.png')
    plot_box(df, 'LABEL', 'HDRFDST', 'Distribution of Hausdorff Distance by Anatomical Structure (lower is better)', 'Hausdorff Distance', save_dir, 'HD_scores_boxplot.png')
    plot_box(df, 'LABEL', 'AVGDIST', 'Distribution of ASD by Anatomical Structure (lower is better)', 'Average Surface Distance', save_dir, 'ASD_scores_boxplot.png')





if __name__ == '__main__':
    main()
