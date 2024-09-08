# markov_checker

This project contains files for a forthcoming arXiv report on checking the Markov condition.

## Installation

Before running this code, the following Python packages need to be installed:

```angular2html
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

In addition, some Java code needs to be installed. This can be done by running the following Python script in this directory:

```
setup_java_and_tetrad.py
```

## Usage

The main Python script is `markov_checker_simulation.py`. This script can be run from the command line with the following arguments:

```angular2html
python markov_checker_simulation.py
```
