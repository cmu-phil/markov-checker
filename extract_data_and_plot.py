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
    df['bic'] = pd.to_numeric(df['bic'], errors='coerce')
    df['|G|'] = pd.to_numeric(df['|G|'], errors='coerce')

    return df

import os

def generate_plot(df, dir_name, file_name, x_var, y_var, kldiv_var='kldiv', edge_var='|G|', alg_var='alg', p_value_var='p_ks', alpha=0.05,
                  output_file='plot.png', transparency=0.7, palette = "hsv"):

    bic_var = 'bic'

    # Convert x_var, y_var, p_value_var, kldiv_var, and edge_var to numeric, coercing errors to NaN
    df[x_var] = pd.to_numeric(df[x_var], errors='coerce')
    df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
    df[p_value_var] = pd.to_numeric(df[p_value_var], errors='coerce')
    df[kldiv_var] = pd.to_numeric(df[kldiv_var], errors='coerce')

    # df["kldiv_log"] = df[kldiv_var].apply(lambda x: np.log(x) if x > 0 else 0)
    # df[kldiv_var] = df["kldiv_log"]

    df[edge_var] = pd.to_numeric(df[edge_var], errors='coerce')

    # if pd contains 'shd' column, convert it to numeric
    if 'shd' in df.columns:
        df['shd'] = pd.to_numeric(df['shd'], errors='coerce')
        df.loc[df['shd'] < 0, 'shd'] = pd.NA

    df[bic_var] = pd.to_numeric(df[bic_var], errors='coerce')

    # Drop rows where x_var, y_var, alg_var, p_value_var, kldiv_var, or edge_var contains NaN
    df_clean = df.dropna(subset=[x_var, y_var, alg_var, p_value_var, kldiv_var, bic_var, edge_var])

    # Filter rows where p_value_var > alpha
    above_alpha_df = df_clean[df_clean[p_value_var] > alpha]

    # Compute the global maximim bic and global minimum |G| across this filtered data
    global_max_bic = above_alpha_df[bic_var].max() if not above_alpha_df.empty else None
    global_min_edges = above_alpha_df[edge_var].min() if not above_alpha_df.empty else None

    # print out the 'alg' and 'param' values for the global global minimum |G|
    if global_min_edges is not None:
        print("CAFS alg and param values for global minimum |G|")
        print(above_alpha_df[above_alpha_df[edge_var] == global_min_edges][[alg_var, 'param']])

    # Find the points where bic is maximized and where |G| is minimized
    kldiv_minimal_points = above_alpha_df[above_alpha_df[bic_var] == global_max_bic] if global_max_bic is not None else pd.DataFrame()
    edge_minimal_points = above_alpha_df[above_alpha_df[edge_var] == global_min_edges] if global_min_edges is not None else pd.DataFrame()

    # Create a color palette for the unique algorithms
    unique_algs = df_clean[alg_var].unique()

    # # temporarily remove 'true' from the list
    # if 'true' in unique_algs:
    #
    # palette = sns.color_palette(palette, len(unique_algs))

    # add 'true' to the end
    if 'true' in df_clean[alg_var].unique():
        unique_algs = [alg for alg in unique_algs if alg != 'true']
        palette = sns.color_palette(palette, len(unique_algs))
        color_map = {alg: palette[i] for i, alg in enumerate(unique_algs)}

        # map 'true' to 'red'
        unique_algs.append('true')
        color_map['true'] = 'red'
    else:
        unique_algs = list(unique_algs)
        palette = sns.color_palette(palette, len(unique_algs))
        color_map = {alg: palette[i] for i, alg in enumerate(unique_algs)}

    # # move 'true' to the end
    # if 'true' in unique_algs:
    #     unique_algs = [alg for alg in unique_algs if alg != 'true']
    #     unique_algs.append('true')
    #
    # palette = sns.color_palette(palette, len(unique_algs) - 1)
    #


    # color_map = {alg: palette[i] for i, alg in enumerate(unique_algs)}
    # color_map['true'] = 'red'

    # Create a simple scatter plot
    plt.figure(figsize=(8, 6))

    # Define markers
    marker_star = '*'  # Star for kldiv minimal point
    marker_below_threshold = 'o'  # Circle for pvalue var <= ALPHA
    larger_marker_size = 120  # Set a larger marker size for general points above threshold
    largest_marker_size = 300  # Set an even larger size for the minimal points

    # Plot each algorithm with its own color
    for alg in unique_algs:
        if alg == 'lingam':  # Skip lingam because it doesn't target the linear Gaussian case.
            continue

        subset = df_clean[df_clean[alg_var] == alg]

        # Separate the points by p_value threshold
        above_threshold = subset[subset[p_value_var] > alpha]
        below_threshold = subset[subset[p_value_var] <= alpha]

        # Plot points where p <= ALPHA with circle markers
        if not below_threshold.empty:
            plt.scatter(below_threshold[x_var], below_threshold[y_var], label=f"{alg} (p ≤ {alpha})",
                        color=color_map[alg], marker=marker_below_threshold, alpha=transparency)

        # Plot points where p > ALPHA with star markers (non-minimal)
        if not above_threshold.empty:
            if not above_threshold.empty:
                plt.scatter(above_threshold[x_var], above_threshold[y_var], label=f"{alg} (p > {alpha})",
                            color=color_map[alg], marker=marker_star, s=larger_marker_size, alpha=transparency)

    # Highlight the edge-minimal points
    if not edge_minimal_points.empty:
        plt.scatter(edge_minimal_points[x_var], edge_minimal_points[y_var], label=f"CAFS",
                    facecolors='none',
                    edgecolors='red',
                    marker=marker_star, s=largest_marker_size, alpha=transparency)

    # Set labels and title
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f'Scatterplot of {y_var} vs {x_var} for {file_name}')

    # Add a legend to show which color corresponds to each algorithm and p-value condition
    plt.legend(title=alg_var)
    plt.xscale('log')
    plt.xlim(1e-4, 1)

    # Show the plot
    plt.tight_layout()

    dir = f"plots/{dir_name}/plots_lg_{x_var}_{y_var}"

    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.savefig(f"{dir}/{output_file}")
    plt.show()



