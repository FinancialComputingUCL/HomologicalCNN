from collections import Counter
from multiprocessing import Pool

import networkx as nx

from tmfg_core import *


class Bootstrapped_Similarity_Matrix:
    def __init__(self, df, correlation_type, number_of_repetitions, parallel=True):

        self.df = pd.DataFrame(df)
        self.correlation_type = correlation_type
        self.original_correlation_matrix = self.get_correlation_matrix(self.df)
        self.number_of_repetitions = number_of_repetitions
        self.parallel = parallel

        _, _, self.adjacency_matrix = TMFG(np.square(self.original_correlation_matrix)).compute_TMFG()
        self.nx_network = nx.from_numpy_matrix(self.adjacency_matrix)

        self.original_network = copy.copy(self.nx_network)

        self.numbered_edges_nx_network = {}
        for e in range(len(self.nx_network.edges())):
            self.numbered_edges_nx_network[e] = 0

    def get_correlation_matrix(self, df):
        try:
            correlation_matrix = df.corr(method=self.correlation_type)
            return correlation_matrix
        except:
            raise Exception("Correlation type for similarity matrix not supported.")

    def average_matrix(self, corr_matrices):
        n = corr_matrices[0].shape[0]
        stacked_matrices = np.stack(corr_matrices, axis=2)
        average_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                row = stacked_matrices[i, j, :]
                average_matrix[i, j] = np.mean(row)
        return average_matrix

    def median_matrix(self, corr_matrices):
        n = corr_matrices[0].shape[0]
        stacked_matrices = np.stack(corr_matrices, axis=2)
        median_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                row = stacked_matrices[i, j, :]
                median_matrix[i, j] = np.median(row)
        return median_matrix

    def compute_bootstrapping(self):
        array_dict = []

        if self.parallel:
            pool = Pool(processes=8)
            results = [pool.apply_async(self.perform_bootstrapping, [self.numbered_edges_nx_network, self.nx_network, self.df]) for _ in range(self.number_of_repetitions)]
            pool.close()

            for r in results:
                array_dict.append(r.get())
        else:
            for _ in range(self.number_of_repetitions):
                a = self.perform_bootstrapping(self.numbered_edges_nx_network, self.nx_network, self.df)
                array_dict.append(a)

        list_of_similarity_matrices = []
        for d in array_dict:
            list_of_similarity_matrices.append(d[3])

        median_similarity_matrix = self.average_matrix(list_of_similarity_matrices) #self.median_matrix(list_of_similarity_matrices) #TODO: uncomment if you want to use median instead of average.

        cliques, separators, local_adjacency_matrix = TMFG(np.square(median_similarity_matrix)).compute_TMFG()
        self.nx_network = nx.from_numpy_matrix(local_adjacency_matrix)
        return cliques, separators, self.nx_network

    def perform_bootstrapping(self, numbered_edges, g, random_matrix):
        numbered_edges = numbered_edges.copy()
        bootstrapped = random_matrix.copy().sample(n=len(random_matrix), replace=True).reset_index(drop=True)
        corr_matrix_bootstrapped = self.get_correlation_matrix(bootstrapped)

        _, _, adjacency_matrix = TMFG(corr_matrix_bootstrapped).compute_TMFG()
        bootstrapped_network = nx.from_numpy_matrix(adjacency_matrix)

        for e in enumerate(g.edges()):
            if bootstrapped_network.has_edge(e[1][0], e[1][1]):
                numbered_edges[e[0]] += 1

        return numbered_edges, bootstrapped_network, bootstrapped_network.degree(), corr_matrix_bootstrapped
