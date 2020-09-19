#!/usr/bin/python
# ---------------------------------------------------------------------------

"""
Run target (phenotype) permutation. The phenotype is permuted across samples.

	usage: target_permute_NETPHIX.py [-h] [-sol ILP_SOL_FILE] [-map GENE_MAP_FILE]
									 [-idx INDEX_NAME] [-sep SEP]
									 [-max_time MAX_TIME] [-add_file ADD_FILE]
									 alt_file target_file correlation k
									 filter_high filter_low density norm penalty
									 netfile_name permutations permutefile_name

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
	  netfile_name          network file
	  permutations          num of permutations
	  permutefile_name      file name to write the permutation results

	optional arguments:
	  -h, --help            show this help message and exit
	  -sol ILP_SOL_FILE, --ILP_sol_file ILP_SOL_FILE
							file with optimal ILP solution with target permutation
							pvalue.
	  -map GENE_MAP_FILE, --gene_map_file GENE_MAP_FILE
							file with ENSG ID to gene name mapping
	  -idx INDEX_NAME, --index_name INDEX_NAME
							index name to use in the target file
	  -sep SEP, --sep SEP   separator in file (\t for default)
	  -max_time MAX_TIME, --max_time MAX_TIME
							time limit for CPLEX
	  -add_file ADD_FILE, --add_file ADD_FILE
							additional mutation file. take OR with the original
							mutations

"""

import cplex
from cplex.exceptions.errors import CplexSolverError
import pandas as pd
import time
import sys
import os
import random
import networkx as nx
import argparse
import numpy as np

import netphix_utils as netphix
import permute_utils as perm

timer_start = time.time()

# fixed params
num_itr = 1                             # NETPHIX finds multiple modules iteratively if num_itr > 1
min_permute = 20                        # minimum # permutations for dynamic test
min_pv = 0.25                            # stop permutations if pv > 0.5 after 100 runs
# penalty = "p"                           # use penalty option for mutual exclusivity)
# CPLEX parameters
num_thread = 16
probe_param = 3		# 3: aggressive probe 1: default
file_param = 3 		# 3: node file on disk, compressed 1: default
# memory_th = 128
treemem_th = 1024 * 10
feasible = 1 		# 1: Emphasize feasibility over optimality (default 0: balanced)
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
parser.add_argument("netfile_name", type=str, help="network file", default= "../data/HumanStringNet.txt")
parser.add_argument("permutations", type=int, help="num of permutations")
parser.add_argument("permutefile_name", help="file name to write the permutation results")
parser.add_argument("-sol", "--ILP_sol_file", type=str,
                    help="file with optimal ILP solution with target permutation pvalue.")
parser.add_argument("-map", "--gene_map_file",
                    help="file with ENSG ID to gene name mapping")
parser.add_argument("-idx", "--index_name",
                    help="index name to use in the target file")
parser.add_argument("-sep", "--sep",
                    help="separator in file (\\t for default)")
parser.add_argument("-max_time", "--max_time", type=float,
                    help="time limit for CPLEX")
parser.add_argument("-add_file", "--add_file",
                    help="additional mutation file. take OR with the original mutations")
# parser.add_argument("-brca_method", "--brca_method",
#                     help="brca inactivation merging method (BRCA or OR)")


args = parser.parse_args()

alteration_file = args.alt_file
target_file = args.target_file
correlation = args.correlation
k = args.k
filter_high = args.filter_high
filter_low = args.filter_low
density = args.density
norm = args.norm
permutations = args.permutations
penalty = args.penalty  # use penalty option to reinforce mutual exclusivity
net_file = args.netfile_name
permutefile_name = args.permutefile_name
solutionfile_name=args.ILP_sol_file
gene_map_file = args.gene_map_file
idx = args.index_name
add_file =  args.add_file
max_time = args.max_time
# brca_method = args.brca_method
if args.sep is not None:
	sep = args.sep
else:
	sep="\t"


#################
# read files
timer_start = time.clock()
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
# run ILP with the original solution

# read solution file if already exists (no need to recompute)
firstline = ""
if (solutionfile_name is not None) and os.path.isfile(solutionfile_name):
	ILP_Solutions, OptCost, pv, firstline = netphix.read_solutionfile(solutionfile_name)
else:
	# Create a new model and populate it below. (no network constraints)
	if penalty == "np":
		print(penalty+" is used...")
		model = netphix.create_ILP_model_np(k, num_samples, num_alterations, num_genes, weights, penalties, gene_mut_lists,
										 alt_df.index, mutated_gene_sets, correlation)
	else:
		model = netphix.create_ILP_model(k, num_samples, num_alterations, num_genes, weights, penalties, gene_mut_lists, alt_df.index, mutated_gene_sets, correlation)

	# add network constraints
	if correlation == "combined2":
		model = netphix.add_sep_density_constraints(model, num_genes, edge_lists, gene_mut_lists, k, density, num_alterations, pos_altered, neg_altered)
	else:
		model = netphix.add_density_constraints(model, num_genes, edge_lists, gene_mut_lists, k, density, num_alterations)
	# set max_time if it is given
	if max_time is not None:
		model.parameters.timelimit.set(max_time)
	# set other cplex parameters to use
	model.parameters.threads.set(num_thread)
	model.parameters.mip.limits.auxrootthreads.set(num_thread)
	model.parameters.mip.strategy.probe.set(probe_param) # agressive probe
	model.parameters.mip.strategy.file.set(file_param) # node file on disk, compressed
	# model.parameters.workmem.set(memory_th)  # working memory limit
	model.parameters.mip.limits.treememory.set(treemem_th)
	model.parameters.workdir.set(work_dir)
	model.parameters.emphasis.memory.set(memory_conserve)
	model.parameters.emphasis.mip.set(feasible)
	model.parameters.mip.strategy.variableselect.set(branch_param)  # strong branching for memory conserving
	model.parameters.mip.cuts.gomory.set(gomory_param) # disable Gomory fractional cuts
	model.parameters.mip.strategy.nodeselect.set(node_param) # best estimate search
	ILP_Solutions = []
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
			timer_end = time.clock() - timer_start
			gap = solution_dic["Gap"]
			solution_dic["time"] = timer_end
			if itr == 0:  # optimal solution
				OptCost = solution_dic["TotCost"]
			ILP_Solutions.append((solution_dic))

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
# permutation test
permuted_weights = [x for x in weights]  # deep copy
PermTotCosts = []

# append results if the file exists
if os.path.isfile(permutefile_name):
	# read file and extract TotCosts
	PermTotCosts += perm.read_permute_file(permutefile_name)
	pstart = len(PermTotCosts)
else: # new file
	pstart = 0
	if permutations > 0:
	    if correlation.startswith("combined") :
	        Label_list = ["selected_pos_muts", "selected_neg_muts", "TotCost", "time", "selected_pos_values", "selected_neg_values"]
	    else:
	        Label_list = ["selected_muts", "TotCost", "time", "selected_values"]
	    params = [target_df.index[0], k, filter_high, filter_low, density, penalty, norm]
	    netphix.write_label(permutefile_name, params, Label_list)

permute = pstart
# for permute in range(pstart, permutations):
while permute < permutations:
	# adaptive test: STOP condition
	pv = (len(list(filter(lambda x: x >= OptCost, PermTotCosts))) + 1) / (len(PermTotCosts) + 1)
	print(str(permute) + "-th iteration pv is " + str(pv))
	if (permute >= min_permute) & (pv > min_pv):  # stop if p-values not significant
		break

	# permute the target
	random.shuffle(permuted_weights)
	mutated_gene_sets, num_alterations, gene_mut_lists, pos_altered, neg_altered, gene_names = netphix.proc_alt(alt_df,
																												permuted_weights, 
																												correlation)
	penalties = netphix.comp_penalties(permuted_weights, num_samples, penalty)

	# Create a new model and populate it below. (no network constraints)
	model = netphix.create_ILP_model(k, num_samples, num_alterations, num_genes, permuted_weights, penalties, gene_mut_lists, alt_df.index,
	                         mutated_gene_sets, correlation)

	# add network constraints
	if correlation == "combined2":
		model = netphix.add_sep_density_constraints(model, num_genes, edge_lists, gene_mut_lists, k, density,
													num_alterations, pos_altered, neg_altered)
	else:
		model = netphix.add_density_constraints(model, num_genes, edge_lists, gene_mut_lists, k, density,
												num_alterations)

	if max_time is not None:
		model.parameters.timelimit.set(max_time)
		# set other cplex parameters to use
		model.parameters.threads.set(num_thread)
		model.parameters.mip.limits.auxrootthreads.set(num_thread)
		model.parameters.mip.strategy.probe.set(probe_param) # agressive probe
		model.parameters.mip.strategy.file.set(file_param) # node file on disk, compressed
		# model.parameters.workmem.set(memory_th)  # working memory limit
		model.parameters.mip.limits.treememory.set(treemem_th)
		model.parameters.workdir.set(work_dir)
		model.parameters.emphasis.mip.set(feasible)
		# permutation test only decides if there is a solution with cost > the optimal for the original instance
		# stop if the value is less than OptCost
		model.parameters.mip.tolerances.lowercutoff.set(OptCost)
		# stop once a feasible solution is found
		model.parameters.mip.limits.solutions.set(1)
	# Solve ILP
	try:
		model.solve()

	except CplexSolverError as e:
		print("Exception raised during solve: ")
		print(e)
		if e.args[2] != cplex.exceptions.error_codes.CPXERR_NO_SOLN:
			continue

	solution = model.solution
	# Display solution.
	solution_dic, selected_idx = netphix.proc_solution(solution, alt_df, k, pos_altered, neg_altered)
	# if solution_dic["Gap"] > gap_th:
	# 	print("gap is "+str(solution_dic["Gap"]) + " -- not optimal..")
	# 	print(str(permute) + "th run again")
	# 	continue
	permute += 1
	timer_end = time.clock() - timer_start
	PermTotCosts.append(solution_dic["TotCost"])

	# write permutation solution (write solutions as computed)
	solution_dic["time"] = timer_end
	netphix.write_solutionline(permutefile_name, solution_dic)

# write optimal solution with pv
if correlation.startswith("combined") :
	Label_list = ["selected_pos_muts", "selected_neg_muts", "TotCost", "time", "selected_pos_values", "selected_neg_values", "pv"]
else:
	Label_list = ["selected_muts", "TotCost", "time", "selected_values", "pv"]


if solutionfile_name is not None:
	if len(firstline) > 0: # use the firstline of the solution file if exists
		params = firstline.split("\t")
	elif idx is None:
		params = [target_df.index[0], k, filter_high, filter_low, density, penalty, norm, num_thread, max_time, gap]
	else:
		params = [idx, k, filter_high, filter_low, density, penalty, norm, num_thread, max_time, gap]
	netphix.write_label(solutionfile_name, params, Label_list)
	for solution_dic in ILP_Solutions:
	    if len(PermTotCosts) > 0:
	        pv = (len(list(filter(lambda x: x >= OptCost, PermTotCosts))) + 1) / (len(PermTotCosts) + 1)
	        print(pv)
	        solution_dic["pv"] = pv
	    netphix.write_solutionline(solutionfile_name, solution_dic)