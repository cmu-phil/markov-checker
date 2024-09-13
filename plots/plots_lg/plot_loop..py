import os
import extract_data_and_plot as ep

directory = "../../alg_output/markov_check_lg/"

for filename in os.listdir(directory):
    if filename.startswith("result") and filename.endswith(".txt"):  # Process only result files
        file_path = os.path.join(directory, filename)
        df = ep.extract_first_dataset(file_path)
        output_file = filename.replace(".txt", ".png")  # Save each plot with a corresponding .png name
        # ep.generate_plot(df, filename, output_file=output_file)
        ep.generate_plot(df, filename, "kldiv", "avgsd", output_file=output_file, transparency=0.7, USE_KS=True)

