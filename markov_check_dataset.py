# Runs the Markov check algorithm and parameter selection (MAPS) on a dataset.
#
# The MAPS algorithm runs a variety of algorithms and parameter choices on a dataset and outputs the results.
# The results include the number of edges in the estimated graph, the number of parameters in the model, the number of
# valid independence tests that were performed for independencies implied by the estimated graph, the p-value of the
# Anderson Darling test of Uniformity, the distance of the p-value of the independence test from alpha, the standard BIC
# score of the estimated graph, the CFI, NFI, and NNFI for a given model using lavaan.

import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = "../py-tetrad/pytetrad"
sys.path.append(BASE_DIR)

# Start the JVM and import the necessary Java packages
import jpype.imports

jpype.startJVM("-Xmx20g", classpath=[f"{BASE_DIR}/resources/tetrad-current.jar"])

import pytetrad_tools.TetradSearch as TetradSearch
import pytetrad_tools.translate as translate
import java.util as util
import edu.cmu.tetrad.search as tetrad_search
import edu.cmu.tetrad.graph as tetrad_graph
from edu.cmu.tetrad.util import Params, Parameters
import edu.cmu.tetrad.algcomparison.independence as independence
import edu.cmu.tetrad.algcomparison.statistic as statistic

from lingam import DirectLiNGAM
from dagma.linear import DagmaLinear

# Import R packages
from rpy2.robjects import ListVector
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import get_conversion
from rpy2.robjects.pandas2ri import converter
from rpy2.robjects.packages import importr

base = importr("base")
lavaan = importr("lavaan")
performance = importr("performance")


class FindGoodModelDataset():

    def __init__(self, location, file=None, data_file=None, num_nodes=5, avg_degree=2, num_latents=0, sample_size=100,
                 sim_type='lg'):
        print("FindGoodModel", "location", location)

        self.file = None
        self.sim_type = None

        self.location = location

        self.num_starts = 2
        self.alpha = 0.01
        self.percentResample = 1
        self.sim_type = sim_type
        self.sample_size = sample_size

        self.params = Parameters()
        self.params.set(Params.ALPHA, self.alpha)
        self.params.set(Params.NUM_STARTS, self.num_starts)

        self.frac_dep_under_null = 0

        self.base = importr('base')
        self.bidag = importr('BiDAG')
        self.pchc = importr("pchc")

        self.structure_prior = 0

        self.pchc = importr('pchc')
        self.base = importr('base')
        self.bidag = importr('BiDAG')

        self.train = None
        self.test = None

        self.file = file

        self.data_file = data_file

        # Load the data
        if data_file == None:
            raise Exception('data_file is required')

        data = pd.read_csv(data_file, sep="\t")
        data = data.astype({col: "float64" for col in data.columns})

        self.num_nodes = data.shape[1]
        self.sample_size = data.shape[0]

        print(data)

        self.train = data
        self.test = data

        # self.train, self.test = train_test_split(data, test_size=.5)  # , random_state=42)

        self.train_java = translate.pandas_data_to_tetrad(self.train)
        self.train_numpy = self.train.to_numpy()
        self.nodes = self.train_java.getVariables()

        # self.graph = graph
        # self.sem_im = sem_im

        self.dagma_l1 = 0.03
        self.dagma_w = 0.1
        self.dagma_T = 5

        self.mmhc_max_k = 10

        self.mmhc_starts = 10
        self.pchc_starts = 10

    # This script calculates the CFI, NFI, and NNFI for a given model using lavaan.
    def get_stats(self, df, graph):
        dag = tetrad_graph.GraphTransforms.dagFromCpdag(graph)
        model = str(tetrad_graph.GraphSaveLoadUtils.graphToLavaan(dag))
        with (default_converter + converter).context():
            r_df = get_conversion().py2rpy(df)

        fit = lavaan.lavaan(model, data=r_df)
        perf = performance.model_performance(fit)

        return {col: perf.rx(i + 1)[0][0] for i, col in enumerate(perf.colnames)}

    def save_lines(self, alg, params):
        for param in params:
            cpdag, p_ad, fd_indep, edges, line, cpdag, data_java = self.table_line(alg, param)
            self.my_print(line)

    def print_info(self, msg):
        self.my_print()
        self.my_print(msg)
        self.my_print()

    def print_parameter_defs(self):
        self.my_print('THE FOLLOWING CAN BE GIVEN WITHOUT KNOWING THE GROUND TRUTH:')
        self.my_print()
        self.my_print('alg = the chosen algorithm')
        self.my_print("param = the parameter that's being varied (only one for this script)")
        self.my_print('nodes = # of measured nodes in the true graph')
        self.my_print(
            'cpdag = 1 if the result is a CPDAG, 0 if not')
        self.my_print('|G| = # edges in the estimated graph')
        self.my_print('num_params = the number of parameters in the model')
        self.my_print(
            'numind = the number of valid independence tests that were performed for independencies implied by the estimated graph')
        self.my_print('p_ad = p-value of the Anderson Darling test of Uniformity')
        self.my_print(f'|alpha| = distance of the p-value of the independence test from alpha = {self.alpha}')
        self.my_print('bic = the standard BIC score of the estimated graph')
        self.my_print('edges = # edges in the estimated graph')
        self.my_print(f'sample size = {self.sample_size}')
        self.my_print()
        self.my_print('THE FOLLOWING REQUIRE KNOWING THE GROUND TRUTH:')
        self.my_print()
        self.my_print('|G*| = # edges in the true graph')
        self.my_print('ap = adjacency precision')
        self.my_print('ar = adjacency recall')
        self.my_print('ahp = arrowhead precision')
        self.my_print('ahr = arrowhead recall')
        self.my_print('f1 = adjacency F1 score')
        self.my_print('f0.5 = adjacency F0.5 score')
        self.my_print('f2 = adjacency F2 score')
        self.my_print()

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    # def get_graph(self):
    #     return self.graph

    # def get_sem_im(self):
    #     return self.sem_im

    def print_lines(self, lines):
        self.header()
        for _line in lines:
            self.my_print(_line)

    def my_print(self, str=''):
        print(str, file=self.file, flush=True)
        print(str, flush=True)

    def table_line(self, alg, param):
        graph = self.get_model(alg, param)

        data_java = translate.pandas_data_to_tetrad(self.test)

        bic, num_params = self.accuracy(graph, data_java)

        test_java = translate.pandas_data_to_tetrad(self.test)

        cpdag, a2Star, p_ad, p_ks, kl_div, frac_dep_null, num_test_indep, num_test_dep \
            = self.markov_check(graph, test_java, self.params)

        dag = tetrad_graph.GraphTransforms.dagFromCpdag(graph)

        stats = self.get_stats(self.test, dag)
        cfi = stats["CFI"]
        nfi = stats["NFI"]
        nnfi = stats["NNFI"]
        chisq = stats["Chi2"]
        dof = stats["Chi2_df"]
        likelihood = stats["Loglikelihood"]
        p_value = stats["p_Chi2"]

        edges = graph.getNumEdges()

        dist_alpha = abs(frac_dep_null - self.alpha)

        line = (f"{alg:14} {param:8.3f}  {self.num_nodes:5}    {edges:3}    {num_params:7.0f}"
                f" {cpdag:6} {num_test_indep:9} "
                f" {a2Star:8.4f} {p_ad:8.4f} {p_ks:8.4f} {kl_div:8.4f}  "
                f" {dist_alpha:7.4f}  {bic:12.4f} {cfi:6.4f}  {nfi:6.4f}  {nnfi:6.4f}"
                f"  {chisq:6.4f}  {dof:6.4f}  {p_value:6.4f} {likelihood:8.4f}")

        return graph, p_ad, frac_dep_null, edges, line, graph, data_java

    def header(self):
        str = (
            f"alg               param  nodes    |G| num_params  cpdag    numind       a2*     p_ad     p_ks    kldiv   |alpha|"
            f"           bic    cfi     nfi    nnfi   chisq     dof  pvalue   loglik")
        self.my_print(str)
        self.my_print('-' * len(str))

        # paramValue is a range of values for the parameter being used. For score-based
        # algorithms it will be penalty discount; for constraint-based it will be alpha.
        # def get_model(self, alg, paramValue):
        #     return tetrad_graph.EdgeListGraph()

        # Could also use pchc::bnmat(a$dag)

    def pchc_graph(self, a, nodes):
        dag = a.rx2('dag')
        graph = tetrad_graph.EdgeListGraph(nodes)

        try:
            arcs = dag.rx2('arcs')
            half = int(len(arcs) / 2)

            for i in range(0, half):
                x = arcs[i]
                y = arcs[i + half]
                graph.addDirectedEdge(nodes.get(self.index(x)), nodes.get(self.index(y)))
        except Exception:
            print('Arcs not available.')

        cpdag = tetrad_graph.GraphTransforms.dagToCpdag(graph)
        return cpdag

    def index(self, variable_name):
        import re

        # Extracting digits from the string
        digits = re.findall(r'\d+', variable_name)

        # Convert the first group of digits to integer
        return int(digits[0]) - 1 if digits else None

    def accuracy(self, est_graph, data):
        est_comparison_graph = tetrad_graph.GraphTransforms.dagToCpdag(est_graph)
        bic = statistic.BicEst().getValue(None, est_comparison_graph, data)
        num_params = statistic.NumParametersEst().getValue(None, est_comparison_graph, data)
        return bic, num_params

    def markov_check(self, graph, data, params):
        cpdag = self.cpdag(graph)

        if self.sim_type == 'mn':
            test = independence.ChiSquare().getTest(data, params)
            test.setMinCountPerCell(1)
        else:
            test = independence.FisherZ().getTest(data, params)

        mc = tetrad_search.MarkovCheck(graph, test, tetrad_search.ConditioningSetType.ORDERED_LOCAL_MARKOV)
        mc.setPercentResample(self.percentResample)
        mc.generateResults(True)
        a2Star = mc.getAndersonDarlingA2Star(True)
        p_ad = mc.getAndersonDarlingP(True)
        p_ks = mc.getKsPValue(True)
        fd_indep = mc.getFractionDependent(True)
        num_tests_indep = mc.getNumTests(True)
        num_test_dep = mc.getNumTests(False)
        results = mc.getResults(True)
        p_values = mc.getPValues(results)

        # Calculate KL-divergence
        bins = 20

        dist = np.histogram(p_values, bins)[0] / len(p_values)

        # Different fromm uniform?
        unif = np.array([1 / bins for _ in range(bins)])

        kldiv = np.mean(dist * np.log(np.clip(dist, 1e-6, 1) / unif))  # dist could be 0 :-(

        return cpdag, a2Star, p_ad, p_ks, kldiv, fd_indep, num_tests_indep, num_test_dep

    def construct_graph(self, g, nodes, cpdag=True):
        graph = tetrad_graph.EdgeListGraph(nodes)
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if g[i, j]: graph.addDirectedEdge(b, a)
        if cpdag: graph = tetrad_graph.GraphTransforms.dagToCpdag(graph)
        return graph

    def bnl_to_tetrad(self, bnl, cpdag=True):
        idx = {f"X{i + 1}": i for i in range(len(self.nodes))}
        num_edges = len(bnl) // 2
        edges = [(bnl[i], bnl[i + num_edges]) for i in range(num_edges)]
        graph = tetrad_graph.EdgeListGraph(self.nodes)
        for edge in edges:
            graph.addDirectedEdge(self.nodes[idx[edge[0]]], self.nodes[idx[edge[1]]])
        if cpdag: tetrad_graph.GraphTransforms.dagToCpdag(graph)
        return graph

    def get_model(self, alg, paramValue):
        _search = TetradSearch.TetradSearch(self.train)
        _search.set_verbose(False)
        _search.use_sem_bic(penalty_discount=paramValue)

        nodes = util.ArrayList()

        for col in self.train.columns:
            nodes.add(tetrad_graph.GraphNode(col))

        # Continuous algorithms
        if alg == 'fges':
            _search.use_sem_bic(penalty_discount=paramValue)
            _search.run_fges(faithfulness_assumed=False)
        elif alg == 'boss':
            _search.use_sem_bic(penalty_discount=paramValue)
            _search.run_boss()
        elif alg == 'grasp':
            _search.use_sem_bic(penalty_discount=paramValue)
            _search.use_fisher_z(0.05)
            _search.run_grasp()
        elif alg == 'sp':
            _search.use_sem_bic(penalty_discount=paramValue)
            _search.run_sp()
        elif alg == 'pc':
            _search.use_fisher_z(paramValue)
            _search.run_pc()
        elif alg == 'cpc':
            _search.use_fisher_z(paramValue)
            _search.run_cpc()
        elif alg == 'lingam':
            dlingam = DirectLiNGAM()
            dlingam.fit(self.train)
            W = dlingam.adjacency_matrix_
            return self.construct_graph(W, nodes, True)
        elif alg == 'bidag':
            bge = self.bidag.scoreparameters("bge", numpy2rpy(self.train_numpy), bgepar=ListVector({"am": 1.0}))
            itmcmc = self.bidag.iterativeMCMC(scorepar=bge, softlimit=9, hardlimit=12, alpha=self.alpha,
                                              verbose=False)
            cpdag = self.construct_graph(np.array(self.base.as_matrix(itmcmc[1]), dtype=int).T, nodes, True)
            return cpdag
        elif alg == 'pchc':
            print("pchc")
            bnl = self.pchc.pchc(numpy2rpy(self.train.values), alpha=self.alpha, restart=self.pchc_starts)
            return self.bnl_to_tetrad(bnl[1][2], cpdag=True)
        elif alg == 'mmhc':
            bnl = self.pchc.mmhc(numpy2rpy(self.train.values), max_k=self.mmhc_max_k, alpha=self.alpha,
                                 restart=self.mmhc_starts)
            return self.bnl_to_tetrad(bnl[1][2], cpdag=True)
        elif alg == 'dagma':
            model = DagmaLinear(loss_type='l2')  # create a linear model with least squares loss
            W = model.fit(self.train.to_numpy(), lambda1=paramValue)  # fit the model with L1 reg. (coeff. 0.02)
            return self.construct_graph(W.T, nodes, True)
        else:
            raise Exception('Unrecognized alg name: ' + alg)

        return _search.get_java()

    def cpdag(self, graph):
        return graph.paths().isLegalCpdag()

    # MAPS = Markov Algorithm and Parameter Selection
    def cafs(self):
        dir = f'markov_check_{self.sim_type}'

        penalties = [10.0, 5.0, 4.0, 3, 2.5, 2, 1.75, 1.5, 1.25, 1]
        alphas = [0.001, 0.01, 0.05, 0.1]  # , 0.2]

        # Create the output directory if it does not exist
        if not os.path.exists(f'{self.location}/{dir}'):
            os.makedirs(f'{self.location}/{dir}')

        result_file = f'{self.location}/{dir}/result.txt'

        if os.path.exists(result_file):
            print("result file exists: " + result_file)
            return

        with (open(result_file, 'w') as file,
              open(f'{self.location}/{dir}/train.txt', 'w') as train_file,
              open(f'{self.location}/{dir}/test.txt', 'w') as test_file):
            find = FindGoodModelDataset(self.location, file, data_file=data_file)

            # print parameter defs and header
            find.print_parameter_defs()
            find.header()

            # go through algorithms and parameter choices and save the best lines (print all lines)
            find.save_lines('lingam', [0])
            find.save_lines('dagma', [0.1, 0.2, 0.3])
            # find.save_lines('pc', alphas)
            # find.save_lines('cpc', alphas)
            find.save_lines('fges', penalties)
            find.save_lines('grasp', penalties)
            find.save_lines('boss', penalties)
            find.save_lines('bidag', [0])
            find.save_lines('mmhc', [0])
            find.save_lines('pchc', [0])

            train = translate.pandas_data_to_tetrad(find.get_train())
            test = translate.pandas_data_to_tetrad(find.get_test())

            # get_stats(train, graph)

            print(train, file=train_file)
            print(test, file=test_file)

            file.close()
            train_file.close()
            test_file.close()


output_dir = 'alg_output'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_file = 'resources/uscrime.continuous.txt'

FindGoodModelDataset(output_dir, data_file=data_file, sim_type='data').cafs()
