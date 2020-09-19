
import sys
import random
import pandas as pd
import networkx as nx
import logging
import numpy as np

def read_permute_file(permutefile):
    """
    read permute file and return list of total costs
    :param permutefile: permutation file
    :return: list of total costs from permutations
    """
    lines = open(permutefile).readlines()
    # sys.stdout.write("%d permutations computed\n" %(len(lines)-2))
    tkns = lines[1].split()
    cost_idx = tkns.index("TotCost")
    # print(cost_idx)
    TotCosts = []
    for l in lines[2:]:
	    tkns = l.split("\t")[1:] # remove first indentation
	    # print(tkns)
	    TotCosts.append(float(tkns[cost_idx]))
    return TotCosts

def gene_permute(alt_df):
    """

    :param alt_df:
    :return:
    """
    permuted_alt = alt_df.copy()
    index_labels  = permuted_alt.index.values
    random.shuffle(index_labels)
    permuted_alt.index = index_labels

    return permuted_alt

def bipartite_double_edge_swap(B, genes, samples, nswap=1, max_tries=1e75):
    """A modified version of bipartite_double_edge_swap function from multi Dendrix,
    which is a modified version of double_edge_swap in NetworkX to preserve the bipartite structure of the graph.

    :param B: nx.Graph B(G, S) a bipartite graph
    :param genes: list of genes G
    :param samples: list of samples S
    :param nswap: int, number of double edge swap to perform
    :param max_tries:int, maximum number of attests to swap edges
    :return: nx.Graph, permuted graph
    """
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(B) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")

    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.

    n = 0
    swapcount = 0

    gkeys, gdegrees = zip(*B.degree(genes).items())  # keys, degree for genes
    gcdf = nx.utils.cumulative_distribution(gdegrees)  # cdf of degree for genes

    pkeys, pdegrees = zip(*B.degree(samples).items())  # keys, degree for samples
    pcdf = nx.utils.cumulative_distribution(pdegrees)  # cdf of degree for samples

    while swapcount < nswap:
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        gi = nx.utils.discrete_sequence(1, cdistribution=gcdf)
        pi = nx.utils.discrete_sequence(1, cdistribution=pcdf)

        gene1 = gkeys[gi[0]]  # convert index to label
        sample1 = pkeys[pi[0]]

        sample2 = random.choice(list(B[gene1]))
        gene2 = random.choice(list(B[sample1]))

        # don't create parallel edges
        if (gene1 not in B[sample1]) and (gene2 not in B[sample2]):
            B.add_edge(gene1, sample1)
            B.add_edge(gene2, sample2)

            B.remove_edge(gene1, sample2)
            B.remove_edge(gene2, sample1)
            swapcount += 1
        if n >= max_tries:
            e = ('Maximum number of swap attempts (%s) exceeded ' % n +
                 'before desired swaps achieved (%s).' % nswap)
            raise nx.NetworkXAlgorithmError(e)
        n += 1
        if n % 10000 == 0:
            logging.debug("%d swaps..\n" % n)
    return B

def permute_mut_graph(B, genes, samples, Q=100):
    """Permutes a given mutation profile B(G, S) by performing |E| * Q edge swaps.

    :param B: nx.Graph B(G, S) a bipartite graph
    :param genes: list of genes G
    :param samples: list of samples S
    :param Q: constant multiplier for number Q * | E | of edge swaps to perform (default and suggested value: 100).
    See `Milo et al. (2003) <http://goo.gl/d723i>`_ for details on choosing Q.

    :returns: H: nx.Graph permuted bipartite graph
    """

    H = B.copy()
    bipartite_double_edge_swap(H, genes, samples, nswap=Q * len(B.edges()))
    return H


def construct_alt_graph(alt_df):
    """ given mutation profile between genes and samples,
    create a bipartite graph for each cancer type separately

    :param mut_dic: mut dictionary gene -> list of cover weights for each sample
    :return mut_graph: alteration bipartite graphs  nx.Graph
    """
    B = nx.Graph()
    genes = alt_df.index
    samples = alt_df.columns
    B.add_nodes_from(genes, bipartite="genes")
    B.add_nodes_from(samples, bipartite="samples")
    for gene in genes:
        # mutated samples
        col = alt_df.loc[gene,:]
        alt_df.columns[np.where(col == 1)]
        edges = [(gene, s) for s in alt_df.columns[np.where(col == 1)]]
        B.add_edges_from(edges)
    return B, genes, samples


def construct_alt_from_graph(B, genes, samples):
    """  construct mutation dic from permuted bipartite graphs for all types

    :param  B:   a bipartite graph B(G, S)
                            S is given as sample indices to make it easy to construct the mutation list
    :param  genes: list of genes
    :param  nsamples: number of samples
    :return
                    mut_dic: dict gene -> altered or not for each sample (in the order as in samples)
    """
    data = np.zeros(shape=(len(genes), len(samples)))
    genes_idx = dict([(genes[i], i) for i in range(len(genes))])
    for i in range(len(samples)):
        s = samples[i]
        mutated_gene_idx = [genes_idx[s] for s in B.neighbors(s)]
        #for s in mutated_sample_idx:
        data[mutated_gene_idx, i] = 1
            # new_alt.loc[gene, s] = 1
    new_alt = pd.DataFrame(data.astype(int), index=genes, columns=samples)
    return new_alt


def alt_permute(alt_df):
    """
    create a random alteration table with the same sample/gene frequency
    :param alt_df:
    :return: permuted_alt
    """
    B, genes, samples = construct_alt_graph(alt_df)
    H = permute_mut_graph(B, genes, samples)
    print("permutation is done")
    permuted_alt = construct_alt_from_graph(H, genes, samples)
    return permuted_alt



