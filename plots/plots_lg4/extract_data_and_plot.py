import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ALPHA = 0.05
USE_KS = True

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
    df['avgsd'] = pd.to_numeric(df['avgsd'], errors='coerce')  # Add avgsd column

    return df


def generate_plot(df, file_name, output_file='plot.png', transparency=0.7):
    # Create a color palette for the unique algorithms
    unique_algs = df['alg'].unique()
    palette = sns.color_palette("hsv", len(unique_algs))
    color_map = {alg: palette[i] for i, alg in enumerate(unique_algs)}

    # Choose the column based on the USE_KS parameter
    p_value_column = 'p_ks' if USE_KS else 'p_ad'

    # Convert the selected p-value column to numeric, forcing errors to NaN
    df[p_value_column] = pd.to_numeric(df[p_value_column], errors='coerce')

    # Plot the data
    plt.figure(figsize=(10, 7))  # Increased figure size

    # Define the markers
    marker_above_threshold = '*'  # Star for p > ALPHA
    marker_below_threshold = 'o'  # Circle for p <= ALPHA
    larger_marker_size = 150  # Set a larger marker size for stars

    # Use color coding and different markers based on the selected p-value condition
    for alg in unique_algs:
        subset = df[df['alg'] == alg]

        # Ensure the selected p-value column is numeric and handle NaN
        above_threshold = subset[subset[p_value_column] > ALPHA].dropna(subset=[p_value_column])
        below_threshold = subset[subset[p_value_column] <= ALPHA].dropna(subset=[p_value_column])

        # Plot points where p <= ALPHA with circle markers if there are any
        if not below_threshold.empty:
            plt.scatter(below_threshold['kldiv'], below_threshold['avgsd'], label=f"{alg} ({p_value_column} ≤ {ALPHA})",
                        color=color_map[alg], marker=marker_below_threshold, alpha=transparency)

        # Plot points where p > ALPHA with star markers if there are any
        if not above_threshold.empty:
            plt.scatter(above_threshold['kldiv'], above_threshold['avgsd'], label=f"{alg} ({p_value_column} > {ALPHA})",
                        color=color_map[alg], marker=marker_above_threshold, s=larger_marker_size, alpha=transparency)

    # Label each point with only 'param' without trailing zeros
    for i in range(len(df)):
        param_cleaned = str(df['param'][i]).rstrip('0').rstrip('.')
        plt.text(df['kldiv'][i], df['avgsd'][i], param_cleaned, fontsize=9, color=color_map[df['alg'][i]])

    # Set labels and title
    plot_title = os.path.splitext(os.path.basename(file_name))[0]  # Get the file name without extension
    plt.xlabel('kldiv')
    plt.ylabel('avgsd')
    plt.title(f'Plot of avgsd against kldiv for {plot_title}')

    # Set X and Y axes to scientific notation
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Add a legend to show which color corresponds to each algorithm and marker for p-value condition
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the layout manually to avoid the tight layout warning
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)  # Adjust margins

    # Show plot
    plt.savefig(output_file)
    plt.show()

# # Test case for a single file
# file_path = "../../alg_output/markov_check_lg/result_25_4.txt"
# df = extract_first_dataset(file_path)
#
# print(df)
#
# print(df.columns)
#
# generate_plot(df, "result_25_4")
