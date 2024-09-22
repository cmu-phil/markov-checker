# markov_checker

This project contains files for a forthcoming arXiv report on checking the Markov condition.

## Overview

We provide a method to simulate data from a DAG, run various algorithms on the data to obtain CPDAGs or DAGs, and then check the Markov condition on these CPDAGs or DAGs with respect to the data.

This does not reproduce the results in the paper, but provides code to generate new examples of the same sort. The exact results in the paper are not reproducible because the data is generated randomly and analyzed on different platforms (Java, Python, and R). However the exact simulation results shown in our paper are included in this repository.

The installation procedure is a bit complicated, since the code runs on three different platforms. The main script for running the simulation is 

```
markov_check_simulation.py
```

This is modified to run the algorithms on a specific dataset in the script, 

```
markov_check_dataset.py
```

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

The following software versions were used to generate the results in the paper:

```
platform       aarch64-apple-darwin20      
arch           aarch64                     
os             darwin20                    
system         aarch64, darwin20           
status                                     
major          4                           
minor          3.2                         
year           2023                        
month          10                          
day            31                          
svn rev        85441                       
language       R                           
version.string R version 4.3.2 (2023-10-31)
nickname       Eye Holes  
Python 3.12.1 (v3.12.1:2305ca5144, Dec  7 2023, 17:23:38) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
For Java: amazon-corretto-21-aarch64, Tetrad 7.6.5
```
