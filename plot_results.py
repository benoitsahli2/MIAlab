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
    # plt.show()
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

    print(df)                 # print a summary, the first 5 rows, and the last 5 rows
    # print(df.to_string())     # print all the rows

    save_dir = os.path.dirname(latest_csv)
    plot_box(df, 'LABEL', 'DICE', 'Distribution of DICE Scores by Anatomical Structure (higher is better)', 'DICE Score', save_dir, 'dice_scores_boxplot.png')
    plot_box(df, 'LABEL', 'HDRFDST95', 'Distribution of Hausdorff 95 Scores by Anatomical Structure (lower is better)', 'Hausdorff 95 Score', save_dir, 'hausdorff95_scores_boxplot.png')
    plot_box(df, 'LABEL', 'HDRFDST', 'Distribution of Hausdorff Scores by Anatomical Structure (lower is better)', 'Hausdorff Score', save_dir, 'hausdorff_scores_boxplot.png')
    plot_box(df, 'LABEL', 'SNSVTY', 'Distribution of Sensitivity Scores by Anatomical Structure (higher is better?)', 'Sensitivity Score', save_dir, 'sensitivity_scores_boxplot.png')
    plot_box(df, 'LABEL', 'PRCISON', 'Distribution of Precision Scores by Anatomical Structure (higher is better?)', 'Precision Score', save_dir, 'precision_scores_boxplot.png')
    plot_box(df, 'LABEL', 'VOLSMTY', 'Distribution of Volume Similarity Scores by Anatomical Structure (higher is better?)', 'Volume Similarity Score', save_dir, 'volume_similarity_scores_boxplot.png')
    plot_box(df, 'LABEL', 'AVGDIST', 'Distribution of Average Surface Distance Scores by Anatomical Structure (lower is better)', 'Average Surface Distance Score', save_dir, 'average_surface_distance_scores_boxplot.png')


if __name__ == '__main__':
    main()
