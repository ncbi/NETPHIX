import cplex
from cplex.exceptions.errors import CplexSolverError
import sys
import random
import pandas as pd
import scipy.stats as ss
import numpy as np
import itertools

# fixed params
max_size = 100  # maximum module size


def preproc_data(target_df, alt_df, filter_high=2, filter_low=-1):
    """
    remove NA from target profile
    keep only common samples in target and alterations
    filter genes based on mutation frequency

    :param target_df: target profile
    :param alt_df: alteration matrix
    :param filter_high: mutation frequency upperbound (excluding the val)
    :param filter_low: mutation frequency lowerbound (excluding the val)
    :return: target_df, alt_df, samples
    """
    # drop NA
    target_df = target_df.dropna(axis=1)

    # keep common samples only
    samples = list(set(target_df.columns).intersection(alt_df.columns))
    num_samples = len(samples)
    alt_df = alt_df.loc[:, samples]
    target_df = target_df.loc[:, samples]

    # filtering with mut frequency
    column_sum = alt_df.sum(axis=1)
    Count_Col = alt_df.shape[1]
    up = Count_Col * filter_high
    down = Count_Col * filter_low
    alt_df = alt_df[((column_sum < up) & (column_sum > down))]

    return target_df, alt_df, samples, num_samples


def norm_target(target_df, norm, correlation):
    """
    normalize target profile and take negative if correlation is negative
    :param target_df: target profile
    :param norm:  normalization method z (z score) or zlog (zscore of log10)
    :param correlation: negative or positive or combined
    :return: norm_target_df
    """
    # normalize
    norm_target_df = pd.DataFrame()
    if norm == "z":
        norm_target_df = target_df.apply(ss.zscore, axis=1, result_type='broadcast')
    elif norm == "zlog":
        norm_target_df = np.log10(target_df + 1).apply(ss.zscore, axis=1, result_type='broadcast')

    if correlation == 'negative':
        norm_target_df = -norm_target_df

    return norm_target_df


def proc_alt(alt_df, weights, corr):
    """
    create list each way (sample-> mutated genes, gene-> mutated samples)
    :param alt_df: alteration matrix
    :param weights: target normalized weights
    :param corr: pos and neg combined combined2
    :return: mutated_gene_sets, num_alterations, pos_altered(P), neg_altered(N)

    """
    mutated_sample_sets = []  # mutated samples for each gene
    pos_alts = []  # genes with positive overall response to the drug P
    neg_alts = []  # genes with negative overall response to the drug N
    num_alterations = alt_df.shape[0]

    for i in range(num_alterations):
        row = alt_df.iloc[i].values
        mutated_samples = np.where(row == 1)[0]
        mutated_sample_sets.append(mutated_samples)
        sum_weights = sum([weights[j] for j in mutated_samples])
        if (corr=="combined") | (corr=="combined2"):
            if sum_weights >= 0:
                pos_alts.append(i)
            else:
                neg_alts.append(i)
        else:  # only use positive side if not combined
            pos_alts.append(i)

    # create a mut_gene list dict = {(gene_idx, idxs of mutations for the gene)}
    alterations = alt_df.index
    genes = list(set([g.split("_")[0] for g in alterations]))
    num_genes = len(genes)
    gene_idx = dict([(genes[j], j) for j in range(len(genes))])
    gene_mut_lists = {}
    if corr == "combined2":   ## separate model
        gene_mut_lists["pos"] = {}
        gene_mut_lists["neg"] = {}
        for j in range(len(genes)):
            gene_mut_lists["pos"][j] = []
            gene_mut_lists["neg"][j] = []
        # add each alteration into the corresponding genes (pos or neg)
        for i in range(num_alterations):
            gene = alterations[i].split("_")[0]
            if gene not in gene_idx:
                continue
            if (corr!="combined2") | (i in pos_alts):
                gene_mut_lists["pos"][gene_idx[gene]].append(i)
            else:
                gene_mut_lists["neg"][gene_idx[gene]].append(i)
    else:
        for j in range(num_genes):
            gene_mut_lists[j] = []
        for i in range(num_alterations):
            gene = alterations[i].split("_")[0]
            if gene not in gene_idx:
                continue
            gene_mut_lists[gene_idx[gene]].append(i)

    # add each alteration into each sample if they are mutated in the sample
    mutated_gene_sets = {}
    mutated_gene_sets["pos"] = []  # mutated genes with positive response for each sample
    mutated_gene_sets["neg"] = []  # mutated genes with negative response for each sample
    # for each sample
    for i in range(alt_df.shape[1]):  #
        col = alt_df.iloc[:, i].values
        mutated_genes = np.where(col == 1)[0]
        pos_mutated = list(set(mutated_genes).intersection(pos_alts))
        neg_mutated = list(set(mutated_genes).intersection(neg_alts))
        pos_mutated.sort()
        neg_mutated.sort()
        mutated_gene_sets["pos"].append(pos_mutated)
        mutated_gene_sets["neg"].append(neg_mutated)

    return mutated_gene_sets, num_alterations, gene_mut_lists, pos_alts, neg_alts, genes


def proc_net(genes, all_net):
    """
    create edge_lists (for each gene, list of neighbors)
    :param genes: list of genes
    :param all_net: network
    :return: edge_lists
        num_genes  (len(genes))
    """
    # Create edge list
    net = all_net.subgraph(genes)
    num_genes = len(genes)
    sys.stdout.write("network size: %d nodes and %d edges\n" % (len(net), len(net.edges())))
    gene_idx = dict([(genes[j], j) for j in range(num_genes)])
    edge_lists = {}
    for j in range(num_genes):
        if genes[j] not in net:
            edge_lists[j] = []
            continue
        edge_lists[j] = [gene_idx[neigh] for neigh in net.neighbors(genes[j])]

    return edge_lists, num_genes

def comp_penalties(weights, num_samples, penalty, epsilon=10**(-5)):
    """
    compute weights/penalties in the objective function
    :param target_df: target profile
    :param num_samples: number of samples
    :param penalty: p or np or value given for average_j
    :return penalties dict: "pos" and "neg" separately
    """

    print(penalty)
    penalties_pos = []
    penalties_neg = []
    average_pos = np.average(list(filter(lambda w: w > 0, weights)))
    average_neg = -np.average(list(filter(lambda w: w < 0, weights)))
    if penalty == "p":
        sys.stdout.write("\timpose penalty..")
        # average_i = target_df.iloc[0][target_df.iloc[0] > 0].mean()
        for l in range(0, num_samples):
            if weights[l] > 0:
                penalties_pos.append(average_pos)
            else:
                penalties_pos.append(-weights[l])
        for l in range(0, num_samples):
            if weights[l] < 0:
                penalties_neg.append(average_neg)
            else:
                penalties_neg.append(weights[l])
    elif penalty == "np":
        sys.stdout.write("\tno penalty..")
        for l in range(0, num_samples):
            penalties_pos.append(0) # no penalty
            penalties_neg.append(0)  # no penalty
    penalties = {}
    penalties["pos"] = penalties_pos
    penalties["neg"] = penalties_neg
    return penalties


def create_ILP_model(k, num_samples, num_alterations, num_genes, weights, penalties, mut_lists, mutated_genes,
                     mutated_gene_sets, corr, old=True):
    """
    create ILP and populate (no network information)

    :param k: size of module
    :param num_samples: number of patients
    :param num_alterations: number of alterations
    :param num_genes: number of genes (can be different from num_alterations when a gene has different types of alterations
    :param weights: phenotype
    :param penalties: dict, penalty for mutual exclusivity for pos and neg separately
    :param mut_lists (pos / neg)
    :param mutated_genes: a mut_gene list dict = {(gene_idx, idxs of mutations for the gene)}
    :param mutated_gene_sets: dict, mutated genes for each sample (pos and neg separately)
    :param corr: combined, combined2, negative, positive
    :return: ILP model

    """
    # Create a new (empty) model and populate it below.
    model = cplex.Cplex()

    # Add variables
    model.variables.add(names=["z_p" + str(j) for j in range(num_samples)],
                        obj=[weights[j] + penalties["pos"][j] for j in range(num_samples)], lb=[0] * num_samples,
                        ub=[1] * num_samples,
                        types=["B"] * num_samples)

    model.variables.add(names=["z_n" + str(j) for j in range(num_samples)],
                        obj=[-weights[j] + penalties["neg"][j] for j in range(num_samples)], lb=[0] * num_samples,
                        ub=[1] * num_samples,
                        types=["B"] * num_samples)

    model.variables.add(names=["y_p" + str(j) for j in range(num_samples)],
                        obj=[-penalties["pos"][j] for j in range(num_samples)], lb=[0] * num_samples,
                        ub=[100] * num_samples,
                        types=["C"] * num_samples)

    model.variables.add(names=["y_n" + str(j) for j in range(num_samples)],
                        obj=[-penalties["neg"][j] for j in range(num_samples)], lb=[0] * num_samples,
                            ub=[100] * num_samples,
                            types=["C"] * num_samples)

    model.variables.add(names=["x" + str(i) for i in range(num_alterations)], lb=[0] * num_alterations,
                        ub=[1] * num_alterations,
                        types=["B"] * num_alterations)

    # for mutation/gene mapping/module size -- different for combined2
    if corr == "combined2":
        model.variables.add(names=["g_p" + str(i) for i in range(num_genes)], lb=[0] * num_genes,
                        ub=[1] * num_genes,
                        types=["B"] * num_genes)
        model.variables.add(names=["g_n" + str(i) for i in range(num_genes)], lb=[0] * num_genes,
                        ub=[1] * num_genes,
                        types=["B"] * num_genes)
    # module size
        model.variables.add(names=["m_p"], lb=[0], ub=[max_size], types=["I"])
        model.variables.add(names=["m_n"], lb=[0], ub=[max_size], types=["I"])
    else:
        model.variables.add(names=["g" + str(i) for i in range(num_genes)], lb=[0] * num_genes,
                            ub=[1] * num_genes,
                            types=["B"] * num_genes)
        model.variables.add(names=["m"], lb=[0], ub=[max_size], types=["I"])

    # Set the type of each variables
    # for i in range(num_alterations):
    # 		model.variables.set_types("x"+str(i), model.variables.type.binary)
    # for j in range(num_samples):
    #     model.variables.set_types("y" + str(j), model.variables.type.integer)

    # Add constraints!
    sets_constraint = cplex.SparsePair(ind=["x" + str(i) for i in range(num_alterations)],
                                       val=[1.0] * num_alterations)
    # size no more than k
    model.linear_constraints.add(lin_expr=[sets_constraint], senses=["L"], rhs=[k])

    for j in range(num_samples):
        number_constraint = cplex.SparsePair(ind=["y_p" + str(j), "z_p" + str(j)], val=[1.0, -1.0])
        model.linear_constraints.add(lin_expr=[number_constraint], senses=["G"], rhs=[0])
        number_constraint = cplex.SparsePair(ind=["y_n" + str(j), "z_n" + str(j)], val=[1.0, -1.0])
        model.linear_constraints.add(lin_expr=[number_constraint], senses=["G"], rhs=[0])

    for j in range(num_samples):
        number2_constraint = cplex.SparsePair(ind=["y_p" + str(j), "z_p" + str(j)], val=[1.0, -k])
        model.linear_constraints.add(lin_expr=[number2_constraint], senses=["L"], rhs=[0])
        number2_constraint = cplex.SparsePair(ind=["y_n" + str(j), "z_n" + str(j)], val=[1.0, -k])
        model.linear_constraints.add(lin_expr=[number2_constraint], senses=["L"], rhs=[0])


    # loss/LOH and gain cannot be selected at the same time
    for i in range(num_genes):
        gain, loss, loh, am, de = -1, -1, -1, -1, -1
        if corr == "combined2":
            combined_mut_list = mut_lists["pos"][i] + mut_lists["neg"][i]
        else:
            combined_mut_list = mut_lists[i]
        for j in combined_mut_list:
            if mutated_genes[j].endswith("gain"):
                gain = j
            elif mutated_genes[j].endswith("loss"):
                loss = j
            elif mutated_genes[j].endswith("LOH"):
                loh = j
            elif mutated_genes[j].endswith("amp"):
                am = j
            elif mutated_genes[j].endswith("del"):
                de = j
        if (gain != -1) and (loss != -1):
            cnv_constraint = cplex.SparsePair(ind=["x" + str(gain), "x" + str(loss)], val=[1.0, 1.0])
            model.linear_constraints.add(lin_expr=[cnv_constraint], senses=["L"], rhs=[1])
        if (gain != -1) and (loh != -1):
            cnv_constraint = cplex.SparsePair(ind=["x" + str(gain), "x" + str(loh)], val=[1.0, 1.0])
            model.linear_constraints.add(lin_expr=[cnv_constraint], senses=["L"], rhs=[1])
        if (am != -1) and (de != -1):
            cnv_constraint = cplex.SparsePair(ind=["x" + str(am), "x" + str(de)], val=[1.0, 1.0])
            model.linear_constraints.add(lin_expr=[cnv_constraint], senses=["L"], rhs=[1])


    sys.stdout.write("\nadding xy constraints..")
    for j in range(num_samples):
        if j % 100 == 0:
            sys.stdout.write("%d\t" % j)
        index = ["y_p" + str(j)]
        value = [1.0]
        for i in mutated_gene_sets["pos"][j]:
            index.append("x" + str(i))
            value.append(-1.0)
        number3_constraint = cplex.SparsePair(ind=index, val=value)
        model.linear_constraints.add(lin_expr=[number3_constraint], senses=["E"], rhs=[0])
        index = ["y_n" + str(j)]
        value = [1.0]
        for i in mutated_gene_sets["neg"][j]:
            index.append("x" + str(i))
            value.append(-1.0)
        number3_constraint = cplex.SparsePair(ind=index, val=value)
        model.linear_constraints.add(lin_expr=[number3_constraint], senses=["E"], rhs=[0])


    # Our objective is to minimize cost. Fixed and variable costs
    # have been set when variables were created.
    model.objective.set_sense(model.objective.sense.maximize)
    return model

def create_ILP_model_np(k, num_samples, num_alterations, num_genes, weights, penalties, mut_lists, mutated_genes,
                     mutated_gene_sets, corr):
    """
    create ILP and populate (no network information)

    :param k: size of module
    :param num_samples: number of patients
    :param num_alterations: number of alterations
    :param num_genes: number of genes (can be different from num_alterations when a gene has different types of alterations
    :param weights: phenotype
    :param penalties: dict, penalty for mutual exclusivity for pos and neg separately
    :param mut_lists (pos / neg)
    :param mutated_genes: a mut_gene list dict = {(gene_idx, idxs of mutations for the gene)}
    :param mutated_gene_sets: dict, mutated genes for each sample (pos and neg separately)
    :param corr: combined, combined2, negative, positive
    :return: ILP model

    """
    # Create a new (empty) model and populate it below.
    model = cplex.Cplex()

    # Add variables
    model.variables.add(names=["z_p" + str(j) for j in range(num_samples)],
                        obj=[weights[j]  for j in range(num_samples)], lb=[0] * num_samples,
                        ub=[1] * num_samples,
                        types=["B"] * num_samples)

    model.variables.add(names=["z_n" + str(j) for j in range(num_samples)],
                        obj=[-weights[j] for j in range(num_samples)], lb=[0] * num_samples,
                        ub=[1] * num_samples,
                        types=["B"] * num_samples)

    model.variables.add(names=["y_p" + str(j) for j in range(num_samples)],
                        lb=[0] * num_samples, ub=[k] * num_samples,
                        types=["C"] * num_samples)

    model.variables.add(names=["y_n" + str(j) for j in range(num_samples)],
                        lb=[0] * num_samples, ub=[k] * num_samples,
                        types=["C"] * num_samples)

    model.variables.add(names=["x" + str(i) for i in range(num_alterations)], lb=[0] * num_alterations,
                        ub=[1] * num_alterations,
                        types=["B"] * num_alterations)

    # for mutation/gene mapping/module size -- different for combined2
    if corr == "combined2":
        model.variables.add(names=["g_p" + str(i) for i in range(num_genes)], lb=[0] * num_genes,
                        ub=[1] * num_genes,
                        types=["B"] * num_genes)
        model.variables.add(names=["g_n" + str(i) for i in range(num_genes)], lb=[0] * num_genes,
                        ub=[1] * num_genes,
                        types=["B"] * num_genes)
    # module size
        model.variables.add(names=["m_p"], lb=[0], ub=[max_size], types=["I"])
        model.variables.add(names=["m_n"], lb=[0], ub=[max_size], types=["I"])
    else:
        model.variables.add(names=["g" + str(i) for i in range(num_genes)], lb=[0] * num_genes,
                            ub=[1] * num_genes,
                            types=["B"] * num_genes)
        model.variables.add(names=["m"], lb=[0], ub=[max_size], types=["I"])

    # Set the type of each variables
    # for i in range(num_alterations):
    # 		model.variables.set_types("x"+str(i), model.variables.type.binary)
    # for j in range(num_samples):
    #     model.variables.set_types("y" + str(j), model.variables.type.integer)

    # Add constraints!
    sets_constraint = cplex.SparsePair(ind=["x" + str(i) for i in range(num_alterations)],
                                       val=[1.0] * num_alterations)
    # size no more than k
    model.linear_constraints.add(lin_expr=[sets_constraint], senses=["L"], rhs=[k])

    # loss/LOH and gain cannot be selected at the same time
    for i in range(num_genes):
        gain, loss, loh, am, de = -1, -1, -1, -1, -1
        if corr == "combined2":
            combined_mut_list = mut_lists["pos"][i] + mut_lists["neg"][i]
        else:
            combined_mut_list = mut_lists[i]
        for j in combined_mut_list:
            if mutated_genes[j].endswith("gain"):
                gain = j
            elif mutated_genes[j].endswith("loss"):
                loss = j
            elif mutated_genes[j].endswith("LOH"):
                loh = j
            elif mutated_genes[j].endswith("amp"):
                am = j
            elif mutated_genes[j].endswith("del"):
                de = j
        if (gain != -1) and (loss != -1):
            cnv_constraint = cplex.SparsePair(ind=["x" + str(gain), "x" + str(loss)], val=[1.0, 1.0])
            model.linear_constraints.add(lin_expr=[cnv_constraint], senses=["L"], rhs=[1])
        if (gain != -1) and (loh != -1):
            cnv_constraint = cplex.SparsePair(ind=["x" + str(gain), "x" + str(loh)], val=[1.0, 1.0])
            model.linear_constraints.add(lin_expr=[cnv_constraint], senses=["L"], rhs=[1])
        if (am != -1) and (de != -1):
            cnv_constraint = cplex.SparsePair(ind=["x" + str(am), "x" + str(de)], val=[1.0, 1.0])
            model.linear_constraints.add(lin_expr=[cnv_constraint], senses=["L"], rhs=[1])

    sys.stdout.write("\nadding xy constraints..")
    for j in range(num_samples):
        if j % 100 == 0:
            sys.stdout.write("%d\t" % j)
        index = ["y_p" + str(j)]
        value = [1.0]
        for i in mutated_gene_sets["pos"][j]:
            index.append("x" + str(i))
            value.append(-1.0)
        number3_constraint = cplex.SparsePair(ind=index, val=value)
        model.linear_constraints.add(lin_expr=[number3_constraint], senses=["E"], rhs=[0])
        index = ["y_n" + str(j)]
        value = [1.0]
        for i in mutated_gene_sets["neg"][j]:
            index.append("x" + str(i))
            value.append(-1.0)
        number3_constraint = cplex.SparsePair(ind=index, val=value)
        model.linear_constraints.add(lin_expr=[number3_constraint], senses=["E"], rhs=[0])

    sys.stdout.write("\nadding zy constraints..")
    for j in range(num_samples):
        number2_constraint = cplex.SparsePair(ind=["y_p" + str(j), "z_p" + str(j)], val=[1.0, -k])
        model.linear_constraints.add(lin_expr=[number2_constraint], senses=["L"], rhs=[0])
        number2_constraint = cplex.SparsePair(ind=["y_n" + str(j), "z_n" + str(j)], val=[1.0, -k])
        model.linear_constraints.add(lin_expr=[number2_constraint], senses=["L"], rhs=[0])

    sys.stdout.write("\nadding yz constraints..")
    for j in range(num_samples):
        number2_constraint = cplex.SparsePair(ind=["y_p" + str(j), "z_p" + str(j)], val=[1.0, -1.0])
        model.linear_constraints.add(lin_expr=[number2_constraint], senses=["G"], rhs=[0])
        number2_constraint = cplex.SparsePair(ind=["y_n" + str(j), "z_n" + str(j)], val=[1.0, -1.0])
        model.linear_constraints.add(lin_expr=[number2_constraint], senses=["G"], rhs=[0])

    sys.stdout.write("\nadding zx constraints..")
    for j in range(num_samples):
        if j % 100 == 0:
            sys.stdout.write("%d\t" % j)
        for i in mutated_gene_sets["pos"][j]:
            index = ["z_p" + str(j)]
            value = [1.0]
            index.append("x" + str(i))
            value.append(-1.0)
            number4_constraint = cplex.SparsePair(ind=index, val=value)
            model.linear_constraints.add(lin_expr=[number4_constraint], senses=["G"], rhs=[0])
        for i in mutated_gene_sets["neg"][j]:
            index = ["z_n" + str(j)]
            value = [1.0]
            index.append("x" + str(i))
            value.append(-1.0)
            number4_constraint = cplex.SparsePair(ind=index, val=value)
            model.linear_constraints.add(lin_expr=[number4_constraint], senses=["G"], rhs=[0])

    # Our objective is to minimize cost. Fixed and variable costs
    # have been set when variables were created.
    model.objective.set_sense(model.objective.sense.maximize)
    return model


def add_density_constraints(model, num_genes, edge_lists, mut_lists, k, density, num_alterations):
    """
    add density constraints to existing ILP model
    :param model: existing ILP model
    :param num_genes:
    :param edge_lists:
    :param mut_lists:
    :param k: module size
    :param density: density of a module (connectivity)
    :param num_alterations: total number of alterations
    :return: model (updated model)
    """

    # mutation/gene mapping
    # make sure gi is 1 iff one or more xj is 1
    # gi <= sum_{j in mlist(i)}(xj)
    for i in range(num_genes):
        index = ["g" + str(i)]
        value = [1]
        for i1 in mut_lists[i]:
            index.append("x" + str(i1))
            value.append(-1)
        density_constraint = cplex.SparsePair(ind=index, val=value)
        rhs = 0
        model.linear_constraints.add(lin_expr=[density_constraint], senses=["L"], rhs=[rhs])
    # mutation/gene mapping gi >= (xj) {all j in mlist(i)}
    for i in range(num_genes):
        for i1 in mut_lists[i]:
            index = ["g" + str(i)]
            value = [1]
            index.append("x" + str(i1))
            value.append(-1)
            density_constraint = cplex.SparsePair(ind=index, val=value)
            rhs = 0
            model.linear_constraints.add(lin_expr=[density_constraint], senses=["G"], rhs=[rhs])

    index = ["m"]
    value = [1]
    for j in range(num_alterations):
        index.append("x" + str(j))
        value.append(-1)
    size_constraint = cplex.SparsePair(index, value)
    model.linear_constraints.add(lin_expr=[size_constraint], senses=["E"], rhs=[0])

    # density constraints (size less than k)
    sum = (k - 1.0) * density
    count = 0
    sys.stdout.write("\nadding density contraints..")
    for i in range(num_genes):
        if i % 1000 == 0:
            sys.stdout.write("%d\t" % i)
        index = ["g" + str(i)]
        value = [-sum]
        for i1 in edge_lists[i]:
            index.append("g" + str(i1))
            value.append(1.0)
        index.append("m")
        value.append(-density)
        density_constraint = cplex.SparsePair(ind=index, val=value)
        rhs = -(sum + density)
        model.linear_constraints.add(lin_expr=[density_constraint], senses=["G"], rhs=[rhs])
    return model


def add_sep_density_constraints(model, num_genes, edge_lists, mut_lists, k, density, num_alterations, pos_alts, neg_alts):
    """
    add density constraints to existing ILP model
    :param neg_alts:
    :param model: existing ILP model
    :param num_genes:
    :param edge_lists:
    :param mut_lists: dict, mutation types for genes  (pos and neg separately)
    :param k: module size
    :param density: density of a module (connectivity)
    :param num_alterations: total number of alterations
    :param pos_alts: positively associated mutations
    :param neg_alts: negatively associated mutations
    :return: model (updated model)
    """
    # mutation/gene mapping
    # make sure gi is 1 iff one or more xj is 1
    for i in range(num_genes):
        # positive side
        index = ["g_p" + str(i)]
        value = [1]
        # gi <= sum_{j in mlist(i)}(xj)
        for i1 in mut_lists["pos"][i]:
            index.append("x" + str(i1))
            value.append(-1)
        density_constraint = cplex.SparsePair(ind=index, val=value)
        rhs = 0
        model.linear_constraints.add(lin_expr=[density_constraint], senses=["L"], rhs=[rhs])
        # mutation/gene mapping gi >= (xj) {all j in mlist(i)}
        for i1 in mut_lists["pos"][i]:
            index = ["g_p" + str(i)]
            value = [1]
            index.append("x" + str(i1))
            value.append(-1)
            density_constraint = cplex.SparsePair(ind=index, val=value)
            rhs = 0
            model.linear_constraints.add(lin_expr=[density_constraint], senses=["G"], rhs=[rhs])
        # negative side
        index = ["g_n" + str(i)]
        value = [1]
        # gi <= sum_{j in mlist(i)}(xj)
        for i1 in mut_lists["neg"][i]:
            index.append("x" + str(i1))
            value.append(-1)
        density_constraint = cplex.SparsePair(ind=index, val=value)
        rhs = 0
        model.linear_constraints.add(lin_expr=[density_constraint], senses=["L"], rhs=[rhs])
        # mutation/gene mapping gi >= (xj) {all j in mlist(i)}
        for i1 in mut_lists["neg"][i]:
            index = ["g_n" + str(i)]
            value = [1]
            index.append("x" + str(i1))
            value.append(-1)
            density_constraint = cplex.SparsePair(ind=index, val=value)
            rhs = 0
            model.linear_constraints.add(lin_expr=[density_constraint], senses=["G"], rhs=[rhs])

    # density constraints (size less than k)
    sum = (k - 1.0) * density
    count = 0
    sys.stdout.write("\nadding density constraints..")
    # m = sum(xi)
    index_p = ["m_p"]
    value_p = [1]
    index_n = ["m_n"]
    value_n = [1]
    for j in range(num_alterations):
        if j in pos_alts:
            index_p.append("x" + str(j))
            value_p.append(-1)
        elif j in neg_alts:
            index_n.append("x" + str(j))
            value_n.append(-1)
    size_constraint_p = cplex.SparsePair(index_p, value_p)
    model.linear_constraints.add(lin_expr=[size_constraint_p], senses=["E"], rhs=[0])
    size_constraint_n = cplex.SparsePair(index_n, value_n)
    model.linear_constraints.add(lin_expr=[size_constraint_n], senses=["E"], rhs=[0])

    for i in range(num_genes):
        if i % 1000 == 0:
            sys.stdout.write("%d\t" % i)
        # positive side
        index = ["g_p" + str(i)]
        value = [-sum]
        for i1 in edge_lists[i]:
            index.append("g_p" + str(i1))
            value.append(1.0)
        index.append("m_p")
        value.append(-density)
        density_constraint = cplex.SparsePair(ind=index, val=value)
        rhs = -(sum + density)
        model.linear_constraints.add(lin_expr=[density_constraint], senses=["G"], rhs=[rhs])
        # negative side
        index = ["g_n" + str(i)]
        value = [-sum]
        for i1 in edge_lists[i]:
            index.append("g_n" + str(i1))
            value.append(1.0)
        index.append("m_n")
        value.append(-density)
        density_constraint = cplex.SparsePair(ind=index, val=value)
        rhs = -(sum + density)
        model.linear_constraints.add(lin_expr=[density_constraint], senses=["G"], rhs=[rhs])
    return model


def run_bootstrap(alt_df, target_df, num_random):
    """
    perform bootstrap
    :param alt_df: alteration matrix, columns are samples and rows are genes
    :param target_df: target file, columns are samples
    :param num_random: how many to select?
    :return: new_alt_df, new_target_df
    """
    orig_sample_size = target_df.shape[1]
    bootstrap = [random.choice(range(orig_sample_size)) for i in range(num_random)]
    return alt_df.iloc[:, bootstrap], target_df.iloc[:, bootstrap]


def proc_solution(solution, alt_df, k, pos_mutated, neg_mutated):
    """
    given ILP solution, extract necessary information
    :param solution: ILP solution
    :param alt_df: alteration matrix, columns are samples and rows are genes
    :param k: module size
    :param pos_mutated: index of mutated positivley (for the drug)
    :param neg_mutated: index of mutated negatively (for the drug)

    :return: selected_pos_muts, selected_neg_muts, TotCost, selected_idx, selected_values
    """
    # infeasible or stopped with no solution

    status = solution.get_status()
    # 103: infeasible
    # 106: node limit, no sol
    # 108: time limit, no sol
    # 112: tree limit, no sol
    if (status == solution.status.MIP_infeasible) or (status in [106, 108, 112]):
        print(status)
        TotCost = 0
        gap = 0.01
        selected_pos_muts = []
        selected_neg_muts = []
        selected_idx = []
        selected_pos_values = []
        selected_neg_values = []
    else:
        # extract information from solution
        TotCost = solution.get_objective_value()
        num_alterations = alt_df.shape[0]
        Upper = solution.MIP.get_best_objective()
        gap = (Upper-TotCost)/TotCost
        # m = solution.get_values("m_p")
        selected_pos_muts = []
        selected_neg_muts = []
        selected_pos_values = []
        selected_neg_values = []
        selected_idx = []  # for iteration
        for i in range(num_alterations):
            if solution.get_values("x" + str(i)) > 0.5:
                mut_name = alt_df.index[i]
                if i in pos_mutated:
                    selected_pos_muts.append(mut_name)
                    selected_pos_values.append(solution.get_values("x" + str(i)))
                elif i in neg_mutated:
                    selected_neg_muts.append(mut_name)
                    selected_neg_values.append(solution.get_values("x" + str(i)))
                else:
                    sys.stderr.write("warning: not mutated?\n")
                    sys.stderr.write(str(i) + "\n")
                    sys.stderr.write(str(pos_mutated) + "\n")
                    sys.stderr.write(str(neg_mutated) + "\n")
                selected_idx.append(i)
        # display solution
        print("Solution status = ", solution.get_status(), ":", end=' ')
        print(solution.status[solution.get_status()])
        print("Total cost = ", TotCost)
        print("pos: " + ",".join(selected_pos_muts) + "\tneg: " + ",".join(selected_neg_muts))
        # print("size of module is " + str(m_p) + " and "+ str(m_n))

    solution_dic = {}
    solution_dic["selected_pos_muts"] = selected_pos_muts
    solution_dic["selected_neg_muts"] = selected_neg_muts
    solution_dic["TotCost"] = TotCost
    solution_dic["Gap"] = gap
    solution_dic["selected_pos_values"] = selected_pos_values
    solution_dic["selected_neg_values"] = selected_neg_values

    return solution_dic, selected_idx


def check_sanity(solution_dic, permuted_weights, all_net):
    """
    sanity check for the solution
    1) check density constraints are satisfied
    2) check pos/neg are correctly assigned

    :param solution_dic:
    :param permuted_weights:
    :param all_net:
    :return:
    """
    selected_pos = solution_dic["selected_pos_muts"]
    selected_neg = solution_dic["selected_neg_muts"]
    # check density constraints
    selected_pos_genes = [x.split("_")[0] for x in selected_pos]
    selected_neg_genes = [x.split("_")[0] for x in selected_neg]
    # check  pos/neg
    row = alt_df.iloc[i].values
    mutated_samples = np.where(row == 1)[0]
    mutated_sample_sets.append(mutated_samples)
    sum_weights = sum([weights[j] for j in mutated_samples])


def sum_bootstrap_results(solution_dics):
    """
    count # appearance of each gene_mutations in bootstrapping
    :param solutions: list of solution dics from bootsrapping
    :return: count_dic
    """
    all_genes = []
    all_edges = []
    for sdic in solution_dics:
        selected_muts = sdic["selected_muts"]
        all_genes.extend(selected_muts)
        selected_muts.sort()  # order doesn't matter
        all_edges.extend(list(itertools.combinations(selected_muts, 2)))
    gene_set = set(all_genes)
    gene_count_dic = {}
    for gene in gene_set:
        gene_count_dic[gene] = (all_genes.count(gene))
    edge_set = set(all_edges)
    edge_count_dic = {}
    for edge in edge_set:
        edge_count_dic[edge] = (all_edges.count(edge))
    return gene_count_dic, edge_count_dic


def write_solutionline(filename, sdic):
    """
    write a solution in one line
    required: selected_muts, TotCost, time
    optional: selected_values, pv, net_pv, alt_pv
    :param filename:
    :param sdic: solution dic
    :return:
    """
    file = open(filename, 'a')
    for selected in ["selected_muts", "selected_pos_muts", "selected_neg_muts"]:
        if selected in sdic:
            file.write('\t%s' % ",".join(sdic[selected]))
    file.write('\t%f' % sdic["TotCost"])
    file.write('\t%f' % (sdic["time"]))
    for selected_vals in ["selected_values", "selected_pos_values", "selected_neg_values"]:
        if selected_vals in sdic:  # won't write in the new, combined run
            file.write('\t%s' % ",".join([str(v) for v in sdic[selected_vals]]))
    if "pv" in sdic:
        file.write('\t%f' % sdic["pv"])
    if "net_pv" in sdic:
        file.write('\t%f' % sdic["net_pv"])
    if "alt_pv" in sdic:
        file.write('\t%f' % sdic["alt_pv"])
    file.write("\n")
    print("writing solution..")
    file.close()


def read_solutionfile(ILP_file):
    """
    read ILP file and return solutions
    file format:
        required: selected_muts, TotCost, time
        optional: selected_values, pv, net_pv, alt_pv

    :param ILP_file:
    :return: Solutions: list of solution dic
    :return: OptCost: optimal cost
    :return opt_pv: pv (None if not given)

    """
    lines = open(ILP_file).readlines()
    # sys.stderr.write("%s" % lines[0])  # target & params
    labels = lines[1].split()
    sol_index = 2
    # if len(labels) == 0:
    #     labels = lines[2].split()
    #     sol_index += 1
    Solutions = []
    # solution = selected_muts, selected_values, TotCost, pv, net_pv, timer_end
    OptCost = -1
    opt_pv = -1
    for i in range(sol_index, len(lines)):
        solution_dic = {}
        # tkns = tuple(lines[i][1:-1].split("\t"))
        tkns = lines[i][1:-1].split("\t")
        for selected in ["selected_muts", "selected_pos_muts", "selected_neg_muts"]:  # for compatibility
            if selected in labels:
                idx = labels.index(selected)
                solution_dic[selected] = tkns[idx].split(",")
                # for old separate run files
                # if selected == "selected_muts":
                # 	if tkns[idx+1] == "":
                # 		del tkns[idx+1]

        idx = labels.index("TotCost")
        solution_dic["TotCost"] = float(tkns[idx])
        idx = labels.index("time")
        solution_dic["time"] = float(tkns[idx])
        for selected_vals in ["selected_values", "selected_pos_values", "selected_neg_values"]:  # for compatibility
            if selected_vals in labels:
                idx = labels.index(selected_vals)
                if len(tkns[idx]) > 0:
                    solution_dic[selected_vals] = [float(x) for x in tkns[idx].split(",")]
                else:
                    solution_dic[selected_vals] = []
                ## for old separate run files
                # if selected_vals == "selected_values":
                # 	if tkns[idx+1] == "":
                # 		del tkns[idx+1]

        if "pv" in labels:
            idx = labels.index("pv")
            solution_dic["pv"] = float(tkns[idx])
        if "net_pv" in labels:
            idx = labels.index("net_pv")
            solution_dic["net_pv"] = float(tkns[idx])
        if "alt_pv" in labels:
            idx = labels.index("alt_pv")
            solution_dic["alt_pv"] = float(tkns[idx])
        Solutions.append(solution_dic)
        if i == sol_index:
            OptCost = solution_dic["TotCost"]
            if "pv" in labels:
                opt_pv = solution_dic["pv"]

    return Solutions, OptCost, opt_pv, lines[0].strip()


def write_label(filename, params, label_list):
    """
    write parameter in the first row
    label in the second
    """
    file = open(filename, 'w')
    file.write("%s\n" % "\t".join([str(x) for x in params]))
    file.write("\t%s\n" % "\t".join(label_list))
    file.close()








