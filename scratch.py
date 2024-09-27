import matplotlib.pyplot as plt
import numpy as np

import dao as dao


def make_data_cont_dao(num_nodes, avg_deg, sample_size):
    """
     Picks a random graph and generates data from it, using the DaO simulation package
     (Andrews, B., & Kummerfeld, E. (2024). Better Simulations for Validating Causal Discovery
     with the DAG-Adaptation of the Onion Method. arXiv preprint arXiv:2405.13100.)
    :param num_nodes: The number of nodes in the graph.
    :param avg_deg: The average degree of the graph.
    :param num_latents: The number of latent variables in the graph.
    :param sample_size: The number of samples to generate.
    :return: The data, nodes, graph, number of nodes, and average degree.
    """

    p = num_nodes  # number of variables
    ad = avg_deg  # average degree
    n = sample_size  # number of samples

    g = dao.er_dag(p, ad=ad)
    g = dao.sf_out(g)
    g = dao.randomize_graph(g)

    R, B, O = dao.corr(g)

    X = dao.simulate(B, O, n, err=lambda *x: np.random.exponential(x[0], x[1]))
    X = dao.standardize(X)

    C = B

    # Get absolute values of non-zero entries
    non_zero_B = np.abs(B[B != 0])
    non_zero_C = np.abs(C[C != 0])

    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first histogram
    ax1.hist(non_zero_B, bins=10, edgecolor='black')
    ax1.set_title("Histogram of B")
    ax1.set_xlabel("Absolute Value")
    ax1.set_ylabel("Frequency")

    # Plot the second histogram
    ax2.hist(non_zero_C, bins=10, edgecolor='black')
    ax2.set_title("Histogram of C")
    ax2.set_xlabel("Absolute Value")
    ax2.set_ylabel("Frequency")

    # Show the plot
    plt.tight_layout()
    plt.show()


# make_data_cont_dao(25, 5, 1000)
