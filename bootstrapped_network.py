from collections import Counter
from multiprocessing import Pool

import networkx as nx

from tmfg_core import *


class Bootstrapped_Network:
    def __init__(self, df, correlation_type, number_of_repetitions, confidence_level, bootstrapping_side, parallel=True):

        self.df = pd.DataFrame(df)
        self.correlation_type = correlation_type
        self.original_correlation_matrix = self.get_correlation_matrix(self.df)
        self.number_of_repetitions = number_of_repetitions
        self.parallel = parallel
        self.confidence_level = confidence_level
        self.bootstrapping_side = bootstrapping_side

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

        my_dict = Counter()
        freq_dict = {}
        for d in array_dict:
            for key, value in d[0].items():
                my_dict[key] += value
            for key, value in dict(d[2]).items():
                if key in freq_dict.keys():
                    freq_dict[key] += value
                else:
                    freq_dict[key] = value

        greater_confidence = [i for i in list(my_dict.keys()) if my_dict[i] >= ((self.number_of_repetitions / 100) * self.confidence_level)]

        list_of_similarity_matrices = []
        for d in array_dict:
            list_of_similarity_matrices.append(d[3])

        median_similarity_matrix = self.median_matrix(list_of_similarity_matrices)

        intermediate_networks_list = []
        for d in array_dict:
            intermediate_networks_list.append(d[1])

        if self.bootstrapping_side == 'network':
            for n, e in enumerate(self.nx_network.edges()):
                if n not in greater_confidence:
                    self.nx_network.remove_edge(e[0], e[1])
            return self.nx_network

        elif self.bootstrapping_side == 'similarity_matrix':
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
