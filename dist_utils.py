import numpy as np
import pandas as pd
import random
import networkx as nx

def comp_avg_dist(module, all_net):
    """
        return average distance within a module
    """
    if len(module) < 2: # do not count singleton modules
        return 0
    dists = []
    for i in range(len(module)-1):
        g1 = module[i]
        for j in range(i+1, len(module)):
            g2= module[j]
            path = nx.shortest_path(all_net, g1, g2)
        dists.append(len(path)-1)
    return sum(dists)/len(dists)


def comp_avg_bet_dist(module1, module2, all_net):
    """
        return average distance within a module
    """
    if (len(module1) < 1) or (len(module2) < 1): # invalid if one of the modules are empty
        return 0
    dists = []
    for i in range(len(module1)):
        g1 = module1[i]
        for j in range(len(module2)):
            g2= module2[j]
            path = nx.shortest_path(all_net, g1, g2)
            dists.append(len(path)-1)
    return sum(dists)/len(dists)


def split_genes(modules):
    """
        parse module string into a list of genes
    """
    if isinstance(modules, float):
        return []
    return [x.split("_")[0] for x in modules.split(",")]


def check_target_nodes(target, all_nodes):
    """
        check if the target is in the network
        need parsing because targets may have suffices etc.
    """
    genes = []
    if isinstance(target, float) and np.isnan(target):
        return []
    for tg in target.split():
        for tg2 in tg.split(","):
            if tg2 in all_nodes:
                genes.append(tg2)
                continue
            if tg2.startswith("("):
                tg2 = tg2[1:]
                if tg2 in all_nodes:
                    genes.append(tg2)
                    continue
            if tg2.endswith(")"):
                tg2 = tg2[:-1]
                if tg2 in all_nodes:
                    genes.append(tg2)
                    continue
    return genes


def comp_dist(gene, all_net, targets):
    """
    for a given gene, compute the distance to each of the drug targets
    np.nan if the gene is not connected

    :param gene: a gene in a module
    :param all_net:  network
    :param targets: drug targets
    :return: list of distances from the gene to ecah of the drug targets
    """
    dists = []
    for tg in targets:
        if nx.has_path(all_net,gene, tg):
            dists.append(nx.shortest_path_length(all_net, gene, tg))
    if len(dists) == 0:
        return np.nan
    return dists


def comp_mean_dist(module_df, all_net, drug_targets_dic):
    """
    for nephix  and uncover modules with dec and inc
    multiple or zero modules for each drug possible
    :param module_df:
    :param all_net:
    :param drug_targets_dic:
    :return:
    """
    new_rows = []
    for module_idx in range(module_df.shape[0]):
        row = module_df.iloc[module_idx].fillna("")  # in case of nan
        drug = row.drug
        targets = drug_targets_dic[drug]

        dec_module = [x.split("_")[0] for x in filter(lambda x: len(x) > 0, row.dec.split(","))]
        inc_module = [x.split("_")[0] for x in filter(lambda x: len(x) > 0, row.inc.split(","))]
        module = dec_module + inc_module
        no_target_module = set(module).difference(targets)

        dec_mean_dist, dec_dists = comp_avg_dist_target(targets, dec_module, all_net)
        inc_mean_dist, inc_dists = comp_avg_dist_target(targets, inc_module, all_net)
        both_mean_dist, both_dists = comp_avg_dist_target(targets, module, all_net)
        no_target_mean_dist, no_target_dists = comp_avg_dist_target(targets, no_target_module, all_net)

        new_rows.append((dec_mean_dist, inc_mean_dist, both_mean_dist, no_target_mean_dist))

    return pd.DataFrame(data=new_rows, columns=["dec", "inc", "both", "no_target"])


# compute distance for each progeni/random module (one module for each drug)
def comp_mean_dist2(module_df, all_net, targets):
    """

    :param module_df: each column has a ordered list of genes for each drug
    :param all_net: network
    :param targets: drug target
    :return: mean distance df -> [g, i] represent the mean distance from g to targets for drug i
    """
    # compute distance to all targets
    dist_df = pd.DataFrame()
    for id in range(1, 266):
        dist_df[id] = module_df[id].apply(comp_dist, args=(all_net, targets[id],))

    # for each drug, compute mean distance from each gene to the drug targets
    sum_df = pd.DataFrame()
    for id in range(1, 266):
        sum_df[id] = dist_df[id].map(sum, na_action='ignore')
    len_df = pd.DataFrame()
    for id in range(1, 266):
        len_df[id] = dist_df[id].map(len, na_action='ignore')
    mean_dist_df = sum_df.div(len_df)

    return mean_dist_df


# library for computing distances
def comp_avg_dist_target(target_genes, module, all_net):
    """ assume target_genes is not empty (already checked)
        return dists
    """
    module = set(module).intersection(all_net.nodes())
    # if either set is empty, return nan
    if (len(module) == 0) | (len(target_genes) == 0):
        return np.nan, np.nan

    # compute avg distance for each gene to targets
    dists = np.array([np.array(comp_dist(gene, all_net, target_genes)).mean() for gene in module])

    return dists.mean(), dists


# choose k random genes
def choose_random_genes(all_genes, k):
    num_genes = len(all_genes)
    return [all_genes[i] for i in random.sample(range(num_genes), 20)]
