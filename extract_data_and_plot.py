import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_first_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_started = False
    columns = []
    data_lines = []

    # Loop through lines to find column names and data
    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("------") and not data_started:
            # The column names are the line before "------"
            columns = lines[i - 1].strip().split()  # Extract column names
            data_started = True  # Mark that we've passed the header

        elif data_started and len(line) == 0:
            # Stop reading once we reach a blank line after data has started
            break

        elif data_started and not line.startswith("------"):
            # Collect the data lines after "------"
            data_lines.append(line.split())

    # Create a DataFrame with the extracted data
    df = pd.DataFrame(data_lines, columns=columns)

    # Convert appropriate columns to numeric (where possible)
    df['param'] = pd.to_numeric(df['param'], errors='coerce')
    df['kldiv'] = pd.to_numeric(df['kldiv'], errors='coerce')
    df['|G|'] = pd.to_numeric(df['|G|'], errors='coerce')

    return df


import os

import os

import matplotlib.ticker as mticker

import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter

def generate_plot(df, file_name, x_var, y_var, alg_var = 'alg', p_value_var = 'p_ks', alpha = 0.05,
                  output_file='plot.png', transparency=0.7):

    # Convert x_var, y_var, and p_value_var to numeric, coercing errors to NaN
    df[x_var] = pd.to_numeric(df[x_var], errors='coerce')
    df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
    df[p_value_var] = pd.to_numeric(df[p_value_var], errors='coerce')

    # Drop rows where x_var, y_var, or alg_var contains NaN
    df_clean = df.dropna(subset=[x_var, y_var, alg_var, p_value_var])

    # Create a color palette for the unique algorithms
    unique_algs = df_clean[alg_var].unique()
    palette = sns.color_palette("hsv", len(unique_algs))
    color_map = {alg: palette[i] for i, alg in enumerate(unique_algs)}

    # Create a simple scatter plot
    plt.figure(figsize=(8, 6))

    # Define markers
    marker_above_threshold = '*'  # Star for pvalue var > ALPHA
    marker_below_threshold = 'o'  # Circle for pvalue var <= ALPHA
    larger_marker_size = 150  # Set a larger marker size for stars

    # Plot each algorithm with its own color
    for alg in unique_algs:
        if alg == 'lingam': # Skip lingam because it doesn't target the linear Gaussian case.
            continue

        subset = df_clean[df_clean[alg_var] == alg]

        # Separate the points by p_value threshold
        above_threshold = subset[subset[p_value_var] > alpha]
        below_threshold = subset[subset[p_value_var] <= alpha]

        # Plot points where p <= ALPHA with circle markers
        if not below_threshold.empty:
            plt.scatter(below_threshold[x_var], below_threshold[y_var], label=f"{alg} (p ≤ {alpha})",
                        color=color_map[alg], marker=marker_below_threshold, alpha=transparency)

        # Plot points where p > ALPHA with star markers
        if not above_threshold.empty:
            plt.scatter(above_threshold[x_var], above_threshold[y_var], label=f"{alg} (p > {alpha})",
                        color=color_map[alg], marker=marker_above_threshold, s=larger_marker_size, alpha=transparency)

    # Set labels and title
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f'Scatterplot of {y_var} vs {x_var} for {file_name}')

    # Add a legend to show which color corresponds to each algorithm and p value condition
    plt.legend(title=alg_var)

    # Show the plot
    plt.tight_layout()

    if not os.path.exists(f'plots/plots_lg'):
        os.makedirs(f'plots/plots_lg')

    plt.savefig(f"plots/plots_lg/{output_file}")
    plt.show()
# Example usage:
# df = pd.read_csv('your_data.csv')  # Replace with your data source
# simple_scatterplot(df, 'x_column_name', 'y_column_name')


