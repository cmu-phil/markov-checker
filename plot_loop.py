# Description: This script is used to generate plots for all the result files in the alg_output/markov_check_data/ directory.
# The plots are generated using the extract_data_and_plot.py script.

import os
import extract_data_and_plot as ep

directory = "alg_output/markov_check_data/"

for filename in os.listdir(directory):
    if filename.startswith("result") and filename.endswith(".txt"):  # Process only result files
        file_path = os.path.join(directory, filename)
        df = ep.extract_first_dataset(file_path)

        output_file = filename.replace(".txt", ".png")  # Save each plot with a corresponding .png name
        # ep.generate_plot(df, filename, output_file=output_file, alpha = 0.01)
        ep.generate_plot(df, filename, "kldiv", "nfi", output_file=output_file, p_value_var = 'p_ks', transparency=0.7, alpha = 0.2)
        # ep.generate_plot(df, filename, "kldiv", "avgminsd", output_file=output_file, transparency=0.7)
        # ep.generate_plot(df, filename, "kldiv", "avgmaxsd", output_file=output_file)

