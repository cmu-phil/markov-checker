import os
import markov_checker_simulation as mcs

output_dir = 'alg_output'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run the desired simulation--uncomment the desired line
mcs.FindGoodModelContinuous(output_dir, sim_type='lg').maps()
# mcs.FindGoodModelContinuous(output_dir, sim_type='exp').maps()
# mcs.FindGoodModelMultinomial(output_dir).maps()
