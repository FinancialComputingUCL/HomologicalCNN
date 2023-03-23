import random
from collections import Counter
from multiprocessing import Pool

import networkx as nx

from tmfg_core import *


class TMFG_Bootstrapped:
    def __init__(self, df, correlation_type, number_of_repetitions, confidence_level, parallel=True, jacknife=False,
                 jacknife_lower_bound=None, seed=None):

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.correlation_type = correlation_type
        self.df = pd.DataFrame(df)

        correlation_matrix = self.get_correlation_matrix(self.df)

        self.cliques, self.separators, self.adjacency_matrix = TMFG(np.square(correlation_matrix)).compute_TMFG()
        self.nx_tmfg = nx.from_numpy_matrix(self.adjacency_matrix)
        self.original_tmfg = copy.copy(self.nx_tmfg)
        self.number_of_repetitions = number_of_repetitions
        self.parallel = parallel
        self.confidence_level = confidence_level
        self.jacknife = jacknife
        self.jacknife_lower_bound = jacknife_lower_bound

        self.numbered_edges_nx_tmfg = {}
        for e in range(len(self.nx_tmfg.edges())):
            self.numbered_edges_nx_tmfg[e] = 0

    def get_correlation_matrix(self, df):
        correlation_matrix = df.corr(method=self.correlation_type)
        return correlation_matrix

    def compute_tmfg_bootstrapping(self):
        array_dict = []

        if self.parallel:
            pool = Pool(processes=8)
            results = [
                pool.apply_async(self.perform_bootstrapping, [self.numbered_edges_nx_tmfg, self.nx_tmfg, self.df]) for
                _ in range(self.number_of_repetitions)]
            pool.close()

            for r in results:
                array_dict.append(r.get())

        else:
            for _ in range(self.number_of_repetitions):
                array_dict.append(self.perform_bootstrapping(self.numbered_edges_nx_tmfg, self.nx_tmfg, self.df))

        my_dict = Counter()
        for d in array_dict:
            for key, value in d[0].items():
                my_dict[key] += value

        greater_95_tmfg = [i for i in list(my_dict.keys()) if
                           my_dict[i] >= ((self.number_of_repetitions / 100) * self.confidence_level)]

        for n, e in enumerate(self.nx_tmfg.edges()):
            if n not in greater_95_tmfg:
                self.nx_tmfg.remove_edge(e[0], e[1])

        intermediate_tmfgs_list = []
        for d in array_dict:
            intermediate_tmfgs_list.append(d[1])

        return self.cliques, self.separators, self.original_tmfg, intermediate_tmfgs_list, self.nx_tmfg

    def perform_bootstrapping(self, numbered_edges_tmfg, g, random_matrix):
        numbered_edges_tmfg = numbered_edges_tmfg.copy()

        if self.jacknife and self.jacknife_lower_bound is not None:
            bootstrapped = random_matrix.sample(
                n=random.randint(int(self.jacknife_lower_bound * len(random_matrix)), len(random_matrix)),
                replace=False).reset_index(drop=True)
        else:
            bootstrapped = random_matrix.sample(n=len(random_matrix), replace=True).reset_index(drop=True)

        corr_matrix_bootstrapped = self.get_correlation_matrix(bootstrapped)

        cliques, separators, adjacency_matrix = TMFG(np.square(corr_matrix_bootstrapped)).compute_TMFG()
        TMFG_bootstrapped = nx.from_numpy_matrix(adjacency_matrix)

        for e in enumerate(g.edges()):
            if TMFG_bootstrapped.has_edge(e[1][0], e[1][1]):
                numbered_edges_tmfg[e[0]] += 1

        return numbered_edges_tmfg, TMFG_bootstrapped
