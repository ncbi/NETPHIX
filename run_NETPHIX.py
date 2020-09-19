#!/usr/bin/python
# ---------------------------------------------------------------------------

"""
Run NETPHIX.
    usage: run_NETPHIX.py [-h] [-tp TARGET_PERM_FILE] [-np NET_PERM_FILE]
                          [-ap ALT_PERM_FILE] [-map GENE_MAP_FILE]
                          [-idx INDEX_NAME] [-sep SEP]
                          [-restricted RESTRICTED_GENE_FILE] [-max_time MAX_TIME]
                          [-pool POOL] [-add_file ADD_FILE] [--append]
                          [--recompute]
                          alt_file target_file correlation k filter_high
                          filter_low density norm penalty net_file
                          solutionfile_name

    positional arguments:
      alt_file              alteration data file (alteration matrix, see
                            data/Alterataion_data/ directory for an example)
      target_file           target profile data file (target (phenotype) see
                            data/Target_data/ directory for an example)
      correlation           positive, negative, combined (for combined model), or
                            combined2 (for separate model)
      k                     size of modules
      filter_high           upperbound mutation frequency of a gene to be included
      filter_low            lowerbound mutation frequency of a gene to be included
      density               module density
      norm                  target normalization method, z or zlog
      penalty               p (penalty) or np (no penalty), penalty to reinforce
                            mutual exclusivity.
      net_file              network file
      solutionfile_name     file name to write the solution

    optional arguments:
      -h, --help            show this help message and exit
      -tp TARGET_PERM_FILE, --target_perm_file TARGET_PERM_FILE
                            file with target permutation results
      -np NET_PERM_FILE, --net_perm_file NET_PERM_FILE
                            file with net permutation results
      -ap ALT_PERM_FILE, --alt_perm_file ALT_PERM_FILE
                            file with alteration table permutation results
      -map GENE_MAP_FILE, --gene_map_file GENE_MAP_FILE
                            file with ENSG ID to gene name mapping
      -idx INDEX_NAME, --index_name INDEX_NAME
                            index name to use in the target file
      -sep SEP, --sep SEP   separator in file (\t for default)
      -restricted RESTRICTED_GENE_FILE, --restricted_gene_file RESTRICTED_GENE_FILE
                            file containing restricted gene modules. compute the
                            objective only with genes in the restricted module
      -max_time MAX_TIME, --max_time MAX_TIME
                            time limit for CPLEX
      -pool POOL, --pool POOL
                            gap limit for solution pool
      -add_file ADD_FILE, --add_file ADD_FILE
                            additional mutation file. take OR with the original
                            mutations
      --append              add solution to existing file
      --recompute           recompute solution and write to the existing file

"""

import cplex
from cplex.exceptions import CplexSolverError
import pandas as pd
import time
import sys
import networkx as nx
import os
import argparse
import numpy as np

import netphix_utils as netphix
import permute_utils as perm


timer_start = time.time()

# fixed params
num_itr = 1                             # NEPHLIX finds multiple modules iteratively if num_itr > 1

# CPLEX parameters
num_thread = 16
probe_param = 3		# 3: aggressive probe 1: default
file_param = 3 		# 3: node file on disk, compressed 1: default
# memory_th = 128
treemem_th = 1024 * 10
feasible = 0 		# 1: Emphasize feasibility over optimality (default 0: balanced)
memory_conserve = 1  # conserve memory 0: default
gomory_param = 0 	# -1: disable Gomory fractional cuts 0: default
branch_param = 0   # 3: strong branching for memory conserving 0: default
node_param = 1		# 2: best estimate search 1: default
work_dir = "/panfs/pan1.be-md.ncbi.nlm.nih.gov/glioma_data/"
gap_th = 0.01       # solution gap in ILP
epsilon = 0.01
gap = -1  # initiall not defined

#################
# read parameters


parser = argparse.ArgumentParser()
# required arguments
parser.add_argument("alt_file", help="alteration data file (alteration matrix, see data/Alterataion_data/ directory for an example)")
parser.add_argument("target_file", help="target profile data file (target (phenotype) see data/Target_data/ directory for an example)")
parser.add_argument("correlation", help="positive, negative, combined (for combined model), or combined2 (for separate model)")
parser.add_argument("k", type=int, help="size of modules")
parser.add_argument("filter_high", type=float, help="upperbound mutation frequency of a gene to be included")
parser.add_argument("filter_low", type=float, help="lowerbound mutation frequency of a gene to be included")
parser.add_argument("density", type=float, help="module density")
parser.add_argument("norm", help="target normalization method, z or zlog")
parser.add_argument("penalty", type=str,
                    help="p (penalty) or np (no penalty), penalty to reinforce mutual exclusivity.")
parser.add_argument("net_file", type=str, help="network file", default= "../data/HumanStringNet.txt")
parser.add_argument("solutionfile_name", help="file name to write the solution")
# optional arguments
parser.add_argument("-tp", "--target_perm_file",
                    help="file with target permutation results")
parser.add_argument("-np", "--net_perm_file",
                    help="file with net permutation results")
parser.add_argument("-ap", "--alt_perm_file",
                    help="file with alteration table permutation results")
parser.add_argument("-map", "--gene_map_file",
                    help="file with ENSG ID to gene name mapping")
parser.add_argument("-idx", "--index_name",
                    help="index name to use in the target file")
parser.add_argument("-sep", "--sep",
                    help="separator in file (\\t for default)")
parser.add_argument("-restricted", "--restricted_gene_file",
                    help="file containing restricted gene modules. compute the objective only with genes in the restricted module")
parser.add_argument("-max_time", "--max_time", type=float,
                    help="time limit for CPLEX")

parser.add_argument("-pool", "--pool", type=float,
                    help="gap limit for solution pool")

parser.add_argument("-add_file", "--add_file",
                    help="additional mutation file. take OR with the original mutations")
# parser.add_argument("-brca_method", "--brca_method",
#                     help="brca inactivation merging method (BRCA or OR)")
parser.add_argument('--append', action='store_true', help="add solution to existing file")
parser.add_argument('--recompute', action='store_true', help="recompute solution and write to the existing file")
args = parser.parse_args()

alteration_file = args.alt_file
target_file = args.target_file
correlation = args.correlation
k = args.k
filter_high = args.filter_high
filter_low = args.filter_low
density = args.density
norm = args.norm
net_file = args.net_file
penalty = args.penalty  # use penalty option to reinforce mutual exclusivity
solutionfile_name = args.solutionfile_name
target_perm_file = args.target_perm_file
net_perm_file = args.net_perm_file
alt_perm_file = args.alt_perm_file
gene_map_file = args.gene_map_file
idx = args.index_name
append = args.append
recompute = args.recompute
restricted_gene_file = args.restricted_gene_file
add_file =  args.add_file
max_time = args.max_time
pool = args.pool
# brca_method = args.brca_method

if args.sep is not None:
    sep = args.sep
else:
    sep="\t"

#################
# read files
sys.stdout.write("reading target, alteration, net files.. %f\n" %timer_start)
target_df = pd.read_csv(target_file, delimiter=sep, index_col=0)
alt_df = pd.read_csv(alteration_file, delimiter=sep, index_col=0)
all_net = nx.read_edgelist(net_file, data=(('weight',float),))

if add_file is not None:
    brcaness = pd.read_table(add_file, delimiter=sep, index_col=0)
    alt_df = alt_df.combine_first(brcaness)
    alt_df.loc[brcaness.index] = np.maximum(alt_df.loc[brcaness.index], brcaness)

if gene_map_file is not None: # map Ensembl ID to gene name
    map_df = pd.read_table(gene_map_file, sep="\t", index_col=0)
    map_dic = dict(zip(map_df.Ensembl, map_df.Name))
    alt_df = alt_df.rename(map_dic)


if restricted_gene_file is not None:
    restricted_module = open(restricted_gene_file).readlines()[3].split()[1].split(",")
    alt_df = alt_df[alt_df.index.isin(restricted_module)]

#################
# processing data
sys.stdout.write("processing data.. \n" )

target_df, alt_df, samples, num_samples  = netphix.preproc_data(target_df, alt_df, filter_high, filter_low)

# exit if there are genes appearing multiple times
duplicates = alt_df.index[alt_df.index.duplicated()]
if len(duplicates) > 0:
    sys.exit("multiple entries for "+",".join(duplicates))

# normalize
norm_target_df = netphix.norm_target(target_df, norm, correlation)
run_ILP = True
# read solution file if already exists (no need to recompute)
if (append is False) and os.path.isfile(solutionfile_name) and (recompute is False):
        Solution_dics, OptCost, pv, firstline = netphix.read_solutionfile(solutionfile_name)
        if len(Solution_dics) > 0:
            run_ILP = False

if run_ILP is True:
    #################
    # prepare to construct ILP model
    sys.stdout.write("preparing to create cplex model.. \n" )

    if idx is None:
        weights = norm_target_df.iloc[0, :].values
    else:
        weights = norm_target_df.loc[idx, :].values

    mutated_gene_sets, num_alterations, gene_mut_lists, pos_altered, neg_altered, gene_names = netphix.proc_alt(alt_df, weights, correlation)
    edge_lists, num_genes = netphix.proc_net(gene_names, all_net)
    penalties = netphix.comp_penalties(weights, num_samples, penalty)

    #################
    # create and run ILP
    # Create a new model and populate it below. (no network constraints)
    if penalty == "np":
        print(penalty + " is used...")
        model = netphix.create_ILP_model_np(k, num_samples, num_alterations, num_genes, weights, penalties,
                                            gene_mut_lists,
                                            alt_df.index, mutated_gene_sets, correlation)
    else:
        model = netphix.create_ILP_model(k, num_samples, num_alterations, num_genes, weights, penalties, gene_mut_lists, alt_df.index,
                             mutated_gene_sets, correlation)

    # add network constraints
    if correlation == "combined2":
        model = netphix.add_sep_density_constraints(model, num_genes, edge_lists, gene_mut_lists, k, density, num_alterations, pos_altered, neg_altered)
    else:
        model = netphix.add_density_constraints(model, num_genes, edge_lists, gene_mut_lists, k, density, num_alterations)
    # set max_time if it is given
    if max_time is not None:
        model.parameters.timelimit.set(max_time)
    # solution pool
    if pool is not None:
        model.parameters.mip.pool.relgap.set(pool)

    # set other cplex parameters to use
    model.parameters.threads.set(num_thread)
    model.parameters.mip.limits.auxrootthreads.set(num_thread)
    model.parameters.mip.strategy.probe.set(probe_param)  # agressive probe
    model.parameters.mip.strategy.file.set(file_param)  # node file on disk, compressed
    # model.parameters.workmem.set(memory_th)  # working memory limit
    model.parameters.mip.limits.treememory.set(treemem_th)
    model.parameters.workdir.set(work_dir)
    model.parameters.emphasis.memory.set(memory_conserve)
    model.parameters.emphasis.mip.set(feasible)
    model.parameters.mip.strategy.variableselect.set(branch_param)  # strong branching for memory conserving
    model.parameters.mip.cuts.gomory.set(gomory_param)  # disable Gomory fractional cuts
    model.parameters.mip.strategy.nodeselect.set(node_param)  # best estimate search
    Solution_dics = []
    timer_start = time.clock()
    for itr in range(num_itr):
        print("start solving ILP: itr " + str(itr))
        # Solve
        try:
            model.solve()
        except CplexSolverError as e:
            print("Exception raised during solve: ")
            print(e)
        else:
            solution = model.solution
            solution_dic, selected_idx = netphix.proc_solution(solution, alt_df, k, pos_altered, neg_altered)
              # if solution_dic["Gap"] > gap_th:
	         #    print("gap is " + str(solution_dic["Gap"]) + " -- not optimal..")
	         #    print(str(permute) + "th run again")
	         #    continue
            timer_end = time.clock() - timer_start
            if itr == 0:  # optimal solution
                OptCost = solution_dic["TotCost"]
            solution_dic["time"] = timer_end
            Solution_dics.append(solution_dic)
            if len(selected_idx) > k:  # condition violated --> rerun
                itr -= 1
                continue
            if len(selected_idx) == 0:  # if no modules are selected
                break
            # remove the selected nodes, and find the next modules
            selected_nodes_constraint = cplex.SparsePair(ind=["x" + str(i) for i in selected_idx],
                                                         val=[1] * int(len(selected_idx)))
            model.linear_constraints.add(lin_expr=[selected_nodes_constraint], senses=["E"], rhs=[0])

#################
# compute p-values and write the results

if correlation.startswith("combined") :
    Label_list = ["selected_pos_muts", "selected_neg_muts", "TotCost", "time", "selected_pos_values", "selected_neg_values"]
else:
    Label_list = ["selected_muts", "TotCost", "time", "selected_values"]

PermTotCosts = []
if (target_perm_file is not None) and os.path.isfile(target_perm_file):
    # read file and extract TotCosts
    PermTotCosts += perm.read_permute_file(target_perm_file)
    Label_list.append("pv")

NetPermTotCosts = []
if (net_perm_file is not None) and os.path.isfile(net_perm_file):
    # read file and extract TotCosts
    NetPermTotCosts += perm.read_permute_file(net_perm_file)
    Label_list.append("net_pv")

AltPermTotCosts = []
if (alt_perm_file is not None) and os.path.isfile(alt_perm_file):
    # read file and extract TotCosts
    AltPermTotCosts += perm.read_permute_file(alt_perm_file)
    Label_list.append("alt_pv")

# open file
if append: # if append, no label
    file = open(solutionfile_name, 'a')
else: # else write the parameters and labels
    if idx is None:
        params = [target_df.index[0], k, filter_high, filter_low, density, penalty, norm, gap]
    else:
        params = [idx, k, filter_high, filter_low, density, penalty, norm, gap]
    netphix.write_label(solutionfile_name, params, Label_list)

# write the solution
for solution_dic in Solution_dics:
    TotCost = solution_dic["TotCost"]
    if len(PermTotCosts) > 0:
        pv = (len(list(filter(lambda x: x >= TotCost, PermTotCosts)))+1)/(len(PermTotCosts)+1)
        solution_dic["pv"] = pv
    if len(NetPermTotCosts) > 0:
        net_pv = (len(list(filter(lambda x: x >= TotCost, NetPermTotCosts))) + 1) / (len(NetPermTotCosts) + 1)
        solution_dic["net_pv"]  = net_pv
    if len(AltPermTotCosts) > 0:
        alt_pv = (len(list(filter(lambda x: x >= TotCost, AltPermTotCosts))) + 1) / (len(AltPermTotCosts) + 1)
        solution_dic["alt_pv"] = alt_pv
    print(solution_dic)
    netphix.write_solutionline(solutionfile_name, solution_dic)


