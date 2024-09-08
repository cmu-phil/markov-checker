# markov_checker

This project contains files for a forthcoming arXiv report on checking the Markov condition.

## Overview

We provide a method to simulate data from a DAG, run various algorithms on the data to obtain CPDAGs or DAGs, and then check the Markov condition on these CPDAGs or DAGs with respect to the data.

This does not reproduce the results in the paper, but provides code to generate new examples of the same sort. The exact results in the paper are not reproducible because the data is generated randomly and analyzed on different platforms (Java, Python, and R). However the exact simulation results shown in our paper are included in this repository.

The installation procedure is a bit complicated, since as mentioned above, the code runs on three different platforms. The main script is `main.py`, which runs the simulation. The user should uncomment the desired simulation in this script and run it. The simulation results are saved in the `alg_output` directory.

## Installation

Before running this code, the following Python packages need to be installed:

```
pip install pandas
pip install numpy
pip install scikit-learn
pip install JPype1
pip install rpy2
pip install lingam
pip install dagma
pip install requests
```

Also, in R, several packages need to be installed. These are referred to by the Python code but need ot be installed, say, in RStudio:

```R
install.packages("lavaan")
install.packages("performance")
install.packages("BiDAG")
install.packages("pchc")
```

In addition, some Java code needs to be installed. A specific Java JDK needs to be downloaded and untarred, and a specific version of the Tetrad software jar needs to be downloaded. This need to be put in the "inst" directory. This can be done automatically by running the following Python script in this directory:

```
setup_java_and_tetrad.py
```

This script downloads the Tetrad software and sets up the Java environment for the Python code to run. The Tetrad software is used to run a number of algorithms.

## Usage

The main Python script is `markov_checker_simulation.py`. This script can be run from the command line with the following arguments:

```angular2html
python markov_checker_simulation.py
```
