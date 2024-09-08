import markov_checker_simulation as mcs

output_dir = 'alg_output'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run the desired simulation--uncomment the desired line
mcs.FindGoodModelContinuous(output_dir, sim_type='lg').maps()
# FindGoodModelContinuous(output_dir, sim_type='exp').maps()
# FindGoodModelMultinomial(output_dir).maps()
# FindGoodModelAncestral(output_dir).maps()