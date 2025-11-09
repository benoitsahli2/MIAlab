import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import os


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

    # print(df)                 # print the first 5 rows, and the last 5 rows
    # print(df.to_string())     # print all the rows

    # Create a figure with larger size for better visibility
    plt.figure(figsize=(12, 6))

    # Create boxplot using seaborn
    sns.boxplot(x='LABEL', y='DICE', data=df)

    # Rotate x-axis labels for better readability
    # plt.xticks(rotation=45)

    # Add title and labels
    plt.title('Distribution of DICE Scores by Anatomical Structure')
    plt.xlabel('Anatomical Structure')
    plt.ylabel('DICE Score')
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Get the directory of the latest results
    save_dir = os.path.dirname(latest_csv)
    save_path = os.path.join(save_dir, 'dice_scores_boxplot.png')
    
    # Save the figure as PNG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {save_path}")
    
    # Display the plot
    plt.show()

    # 2) Autres métriques si disponibles dans le CSV
    # noms typiques avec pymia:
    metric_columns = {
        'HDRFDST95': "Hausdorff 95 (mm)",
        'SENS': "Sensitivity",
        'PREC': "Precision",
        'VOLSMTY': "Volume similarity",
    }

    for col, pretty_name in metric_columns.items():
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='LABEL', y=col, data=df)
            plt.title(f'{pretty_name} by Anatomical Structure')
            plt.xlabel('Anatomical Structure')
            plt.ylabel(pretty_name)
            plt.tight_layout()
            out_name = f'boxplot_{col}.png'
            plt.savefig(os.path.join(save_dir, out_name), dpi=300, bbox_inches='tight')
            plt.show()
        else:
            # juste pour debug éventuel en console, pas obligatoire
            print(f"Column '{col}' not found in results.csv (skipping).")


if __name__ == '__main__':
    main()
