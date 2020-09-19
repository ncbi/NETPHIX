"""
generate simulation data

usage: gen_simulated.py [-h] avg_pv fraction k itr

positional arguments:
  avg_pv      target deviation
  fraction    fraction of affected samples
  k           module size
  itr         iteration number

optional arguments:
  -h, --help  show this help message and exit
"""

import pandas as pd
import numpy as np
import networkx as nx
import scipy.stats as ss
import math
import argparse

parser = argparse.ArgumentParser()
# required arguments
parser.add_argument("avg_pv", type=float, help="target deviation")
parser.add_argument("fraction", type=float, help="fraction of affected samples")
parser.add_argument("k", type=int, help="module size")
parser.add_argument("itr", type=int, help="iteration number")
parser.add_argument('--combined', action='store_true', help="neg/pos combined")

args = parser.parse_args()
itr = args.itr
avg_pv=args.avg_pv
fraction=args.fraction
k=args.k
combined = args.combined

# planted - parameters: avg, std, m, k
std = 0.5
# avg_pvs = [0.005, 0.01, 0.99, 0.995]  # pvalues to generate avg target for planted modules
# fractions = [0.3, 0.2, 0.1]  # fraction of affected samples
# ks = [3, 4, 5]
data_dir = "/home/kimy3/Projects/NetPhix/NetPhix/data/"
sim_dir = data_dir + "Sim/combined2/"
net_file = data_dir + "HumanStringNet.txt"
alt_file = "/panfs/pan1.be-md.ncbi.nlm.nih.gov/icgc_seq/ICGC_analysis_YooAh/DepMap/AlterationsV2.txt"

# background
alt_df = pd.read_csv(alt_file, delimiter="\t", index_col=0)
all_net = nx.read_edgelist(net_file, data=(('weight', float),))

# cleaning alteration table to only include genes in the network
genes = set([g.split("_")[0] for g in alt_df.index]).intersection(all_net.nodes())
alt_df = alt_df[pd.Index([x[0] for x in alt_df.index.str.split("_")]).isin(genes)]

n_samples = alt_df.shape[1]
target = np.random.normal(0, 1, n_samples)
n_alterations = alt_df.shape[0]


# for avg_pv in avg_pvs:
#     for fraction in fractions:
#         for k in ks:
new_target = target.copy()
new_alt_df = alt_df.copy()
new_net = all_net.copy()
# choose abnormal target distribution N1 = N(avg, std)
avg = ss.norm.ppf(avg_pv)
# choose random m samples (10%?) and assign target value from N1
n_affected_samples = int(n_samples * fraction)
affected_target = np.random.normal(avg, std, n_affected_samples)
affected_samples = np.random.choice(n_samples, n_affected_samples, replace=False)

if combined is True:
	# decide negative/positive for each sample
	while True:
		correlation_list = np.random.choice([-1, 1], n_affected_samples)
		pos_samples = [affected_samples[j] for j in filter(lambda x: correlation_list[x] == 1, range(n_affected_samples))]
		neg_samples = [affected_samples[j] for j in filter(lambda x: correlation_list[x] == -1, range(n_affected_samples))]
		n_pos_samples = len(pos_samples)
		n_neg_samples = len(neg_samples)
		# make sure sufficient number of samples both sides
		if (n_pos_samples >= k) & (n_neg_samples >= k):
			break
else:
	correlation_list = [1 for i in range(n_affected_samples)]

# extension for combined2
# assign negative for neg_samples
for i in range(n_affected_samples):
    new_target[affected_samples[i]] = affected_target[i]*correlation_list[i]

# choose random k genes and add edges and mutations randomly
affected_genes_idxs = np.random.choice(n_alterations, k, replace=False)

# choose neg_genes/pos_genes and add edges separately
if combined is True:
	# decide negative/positive for each gene
	genes_correlation_list = np.random.choice([-1, 1], k)
else:
	genes_correlation_list = [1 for i in range(k)]

pos_genes_idxs = [affected_genes_idxs[j] for j in filter(lambda x: genes_correlation_list[x] == 1, range(k))]
neg_genes_idxs = [affected_genes_idxs[j] for j in filter(lambda x: genes_correlation_list[x] == -1, range(k))]

# add edges
def add_edges(new_net, affected_genes_idxs):
	affected_nodes = [alt_df.index[i].split("_")[0] for i in affected_genes_idxs]
	k = len(affected_genes_idxs)
	n_neighbors = math.ceil(0.5 * (k - 1))
	for node in affected_nodes:
	    other_nodes = list(set(affected_nodes).difference([node]))
	    # select random half
	    neighbors = np.random.choice(other_nodes, n_neighbors, replace=False)
	    new_net.add_edges_from([(node, neighbor) for neighbor in neighbors])
	return new_net

new_net = add_edges(new_net, pos_genes_idxs)
new_net = add_edges(new_net, neg_genes_idxs)

# add mutations
def add_mutations(new_alt_df, affected_samples, affected_genes_idxs):
	n_affected_samples = len(affected_samples)
	n_affected_genes_idxs = len(affected_genes_idxs)
	mutations = np.random.choice(n_affected_genes_idxs, n_affected_samples, replace=True)
	for j in range(n_affected_samples):
	    mutation = mutations[j]
	    # print(new_alt_df.index[affected_genes_idxs[mutation]])
	    new_alt_df.iloc[affected_genes_idxs[mutation], affected_samples[j]] = 1
	    # print(str(affected_genes_idxs[mutation]) +","+str(affected_samples[j]))

	return new_alt_df

# add mutations neg/pos separately
new_alt_df = add_mutations(new_alt_df, pos_samples, pos_genes_idxs)
new_alt_df = add_mutations(new_alt_df, neg_samples, neg_genes_idxs)

# print simulated data
# target
params = [str(avg_pv), str(fraction), str(k), str(itr)]
if combined is True:
	params += ["combined"]
new_target_file=sim_dir + "_".join(["target"] + params) + ".tsv"
module_file = sim_dir + "_".join(["module"] + params) + ".txt"
new_alt_file =sim_dir + "_".join(["alt"] + params) + ".tsv.gz"
new_net_file = sim_dir + "_".join(["net"] + params) + ".net"
print(params)

pd.DataFrame([new_target], columns=alt_df.columns).to_csv(new_target_file, sep="\t")
# new_alt
new_alt_df.to_csv(new_alt_file, compression="gzip", sep="\t")
# new_net
nx.write_edgelist(new_net, new_net_file, data=False)

# module
f = open(module_file, 'w')
f.write("avg_pv\t%f\n" % avg_pv)
f.write("fraction\t%f\n" % fraction)
f.write("k\t%d\n" % k)
if combined is True:
	f.write("positive genes\t%s\n" % (",".join([alt_df.index[i] for i in pos_genes_idxs])))
	f.write("negative genes\t%s\n" % (",".join([alt_df.index[i] for i in neg_genes_idxs])))
else:
	f.write("genes\t%s\n" % (",".join([alt_df.index[i] for i in affected_genes_idxs])))
f.close()
