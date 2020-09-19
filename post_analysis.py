"""
Jupyter notbook file was used to generate the files
"""
import netphix_utils as netphix
import os
import pandas as pd
import numpy as np
import scipy.stats as ss
import cv_utils

def restore_drug_name(firstline):
	tkns = firstline.split("\t")
	drug_name = []
	for i in range(len(tkns)):
		if tkns[i].isdigit():
			if (int(tkns[i]) < 10):
				break
		drug_name.append(tkns[i])
		if drug_name[0] == "Bryostatin":
			drug_name.append("1")
	return " ".join(drug_name)

# read netphix results
def read_drug_solution(filename):
	"""
	read the "combined" solution file and return necessary information
	designed specifically for drug netphix results and a wrapper for "read_solutionfile" in netphix_utils.py

	"""
	if os.path.isfile(filename) is False:
		print(filename +" not exist")
		return 1, 'NA', -1, [], []
	solution_dics, cost, p, firstline  = netphix.read_solutionfile(filename)
	if len(solution_dics) < 1:
		return 1, 'NA', -1, [], []
	drug = restore_drug_name(firstline)  # to take care of the case where drug names are not  printed correctly
	# drug = firstline.split("\t")[0]
	selected_pos_muts = solution_dics[0]["selected_pos_muts"]
	selected_neg_muts = solution_dics[0]["selected_neg_muts"]
	if selected_pos_muts == ['']:
		selected_pos_muts = []
	if selected_neg_muts == ['']:
		selected_neg_muts = []
	return p, drug, cost, selected_pos_muts, selected_neg_muts

# def sol_dir():
# 	if penalty == "p":
# 		sol_dir = result_dir + corr
# 	else:
# 		sol_dir = result_dir + corr + "_" + penalty
# 		name_list.append(penalty)
# 	if density != 0.5:
# 		sol_dir += "_" + str(density)
# 		name_list.append(str(density))
# 	sol_dir += "/" + corr + "_" + str(drug_id) + "/all/"


def read_modules_for_drug(drug_id, result_dir, corrs, max_k, penalty="p", density=0.5, freq=1):
	"""
	read all modules for a given drug
	:param drug_id: file id for drug
	:param result_dir: directory
	:param corrs: combined, combined2
	:param max_k: maximum k
	:param penalty: p or np
	:param freq: 1
	:return: DataFrame - each row includes (drug, k, corr, cost, dec_module, inc_module, pv)
	"""
	columns = ["drug", "k", "method", "TotCost", "dec", "inc", "pv"]
	rows  = []
	for corr in corrs:
		for k in range(1, max_k + 1):
			# file name
			name_list = ["depmap_results", str(drug_id), corr, str(k), str(freq)]
			sol_name = result_dir + "_".join(name_list) + ".txt"
			pv, drug, cost, dec, inc = read_drug_solution(sol_name)
			row = (drug, k, corr, cost, ",".join(dec), ",".join(inc), pv)
			rows.append(row)
	return pd.DataFrame(rows, columns=columns)

def read_all_modules(result_dir, corrs, max_k, penalty="p", density=0.5, freq=1, ids=range(1, 266)):
	"""
	read all netphix modules (combined/separate), assume density = 0.5
	:param result_dir:
	:param corrs:
	:param max_k:
	:param penalty: default "p"
	:param ids: list of drug ids to read
	:return: dataframe
	"""
	columns = ["drug", "k", "method", "TotCost", "dec", "inc", "pv"]
	all_modules_df = pd.DataFrame([], columns = columns)
	for drug_id in ids:
		modules_df = read_modules_for_drug(drug_id, result_dir, corrs, max_k, penalty=penalty, density=density, freq=freq)
		all_modules_df = pd.concat([all_modules_df, modules_df])
	all_modules_df.reset_index(drop=True, inplace=True)
	return all_modules_df

def select_sig_modules_for_drug(modules_df, pv_th):
	"""
	choose all maximal modules for a drug among different method/k combinations with the best p-value (< pv_th)

	:param modules_df: list of candidate modules FOR A DRUG
	:param pv_th: significance p-value cutoff
	:return: selected_module_df
	"""
	pvs = modules_df.pv.values
	dec_list = modules_df.dec.values
	inc_list = modules_df.inc.values
	# merge two sets
	module_list = [set(dec_list[i].split(",")).union(inc_list[i].split(",")) for i in range(len(dec_list))]
	for m in module_list:
		if "" in m:
			m.remove("")
	best_pv = min(pvs)
	if best_pv >= pv_th: # pv should be less than pv_th
		return modules_df.iloc[[]]
	# initialize with all modules with best_pv
	selected_module_idxs = list(filter(lambda j: pvs[j] == best_pv, range(modules_df.shape[0])))
	# find maximal set by pairwise comparison
	new_selected_module_idxs = []
	for i in selected_module_idxs:
		# find all maximal sets
		maximal = True
		for j in selected_module_idxs:
			if i == j:
				continue
			if set(module_list[i]).issubset(module_list[j]):
				if len(module_list[i]) < len(module_list[j]): # i is a proper subset of j
					maximal = False
					break
				elif (len(module_list[i]) == len(module_list[j])) and (i > j): # i is the same as j (keep the first one)
					maximal = False
					break
		if maximal is True:
			new_selected_module_idxs.append(i)
	selected_module_idxs = new_selected_module_idxs
	return modules_df.iloc[selected_module_idxs]

def select_sig_module_best_k(result_dir, method, drug_id, pv_th, cost_th):
	"""
	old function
	increase k and choose the best k with significant p-value
	cost(k+1) < (1+cost_th)*cost(k) or p(k+1) > p(k)
	"""
	pvs = []
	costs = []
	pos_list = []
	neg_list = []
	for k in range(1, 6):
		# read combined2
		sol_name = result_dir+"depmap_results_"+str(drug_id)+"_"+method+"_"+str(k)+"_1.txt"
		x = read_drug_solution(sol_name)
		if len(x) == 1:
			p, drug, cost, selected_pos_muts, selected_neg_muts = [1, 'NA', -1, [], []]
		else:
			p, drug, cost, selected_pos_muts, selected_neg_muts = x
		pvs.append(p)
		costs.append(cost)
		pos_list.append(selected_pos_muts)
		neg_list.append(selected_neg_muts)
	for j in range(1, 5):
		j1 = j-1
		cost_diff = costs[j]-costs[j1]
		if (cost_diff < cost_th*costs[j]) or (pvs[j] > pvs[j-1]):  # improvement is not significant or pv getting worse
			sig_j = j1
			break
		elif j == 4:
			sig_j = j
	k = sig_j+1
	if (pvs[sig_j] < 0.05):
		return drug, k, costs[sig_j], pos_list[sig_j], neg_list[sig_j], pvs[sig_j]
	else:
		return drug, -1, 0, [], [], 1


def cross_val_ind(auc_df, drug, alt_df, dec, inc, method="mut", upper=False):
	"""
	test individual module
	:param auc_df: auc dataframe
	:param drug: drug name
	:param alt_df: gene alt dataframe
	:param dec: list of genes associated with dec. sensitivity
	:param inc: list of genes associated with inc. sensitivity
	:param method: inc/dec/mut/both/dec_inc
		inc: inc vs. not inc
		dec: dec vs. not dec
		mut: dec vs. inc  vs.  not mut
		both: dec vs. inc  vs.  not mut vs. both
		dec_inc: dec vs. inc
	:param for compatibility with old version
	:return: pvalue (1 if drug not exist)
	"""

	drug = cv_utils.ctrp_drug_name(drug)

	if drug not in auc_df.index:
		return 1
	ctrp_auc = auc_df.loc[drug]
	if upper is True:
		dec_module = [x.upper() for x in dec]
		inc_module = [x.upper() for x in inc]
	else:
		dec_module = [x for x in dec]
		inc_module = [x for x in inc]

	#  mutations in decreased module
	dec_cover = alt_df[alt_df.index.isin(dec_module)&~alt_df.index.isin(inc_module)].sum()
	dec_cells = dec_cover[dec_cover > 0].index
	dec_auc = ctrp_auc[ctrp_auc.index.isin(dec_cells)].values
	# mutations in increased module
	inc_cover = alt_df[alt_df.index.isin(inc_module) & ~alt_df.index.isin(dec_module)].sum()
	inc_cells = inc_cover[inc_cover > 0].index
	inc_auc = ctrp_auc[ctrp_auc.index.isin(inc_cells)].values

	if method == "dec":
		not_dec_cells = dec_cover[dec_cover == 0].index
		not_dec_auc = ctrp_auc[ctrp_auc.index.isin(not_dec_cells)].values
		sets = [[x for x in dec_auc if ~np.isnan(x)], [x for x in not_dec_auc if ~np.isnan(x)]]
	elif method == "inc":
		not_inc_cells = inc_cover[inc_cover == 0].index
		not_inc_auc = ctrp_auc[ctrp_auc.index.isin(not_inc_cells)].values
		sets = [[x for x in inc_auc if ~np.isnan(x)], [x for x in not_inc_auc if ~np.isnan(x)]]
	elif method ==  "mut":
		not_dec_cells = dec_cover[dec_cover == 0].index
		not_inc_cells = inc_cover[inc_cover == 0].index
		not_mutated_auc = ctrp_auc[ctrp_auc.index.isin(not_inc_cells)&ctrp_auc.index.isin(not_dec_cells)].values
		sets = [[x for x in dec_auc if ~np.isnan(x)], [x for x in not_mutated_auc if ~np.isnan(x)],
				[x for x in inc_auc if ~np.isnan(x)]]
	elif method ==  "both":
		not_dec_cells = dec_cover[dec_cover == 0].index
		not_inc_cells = inc_cover[inc_cover == 0].index
		not_mutated_auc = ctrp_auc[ctrp_auc.index.isin(not_inc_cells)&ctrp_auc.index.isin(not_dec_cells)].values
		both_mutated_auc = ctrp_auc[ctrp_auc.index.isin(inc_cells)&ctrp_auc.index.isin(dec_cells)].values
		sets = [[x for x in dec_auc if ~np.isnan(x)], [x for x in not_mutated_auc if ~np.isnan(x)],
				[x for x in inc_auc if ~np.isnan(x)], [x for x in both_mutated_auc if ~np.isnan(x)] ]
	elif method ==  "dec_inc":
		sets = [[x for x in dec_auc if ~np.isnan(x)],[x for x in inc_auc if ~np.isnan(x)]]

	sets = list(filter(lambda x: len(x) > 0, sets)) # remove empty sets
	if len(sets) == 0:
		return 1

	anova = ss.f_oneway(*tuple(sets))

	return anova.pvalue


# cross validation
def cross_val(modules_df, ctrp_alt_df, ctrp_auc_df, method="mut", ctrp_drugs=None, upper=False):
	"""
	test method
	:param auc_df: drug auc dataframe
	:param alt_df: gene alt dataframe
	:param ctrp: drugs
	:param dec_muts: genes associated with dec. sensitivity
	:param inc_muts: genes associated with inc. sensitivity
	:param method: inc/dec/mut/both/dec_inc
		inc: inc vs. not inc
		dec: dec vs. not dec
		mut: dec vs. inc  vs.  not mut
		both: dec vs. inc  vs.  not mut vs. both
		dec_inc: dec vs. inc
	:param ctrp_drugs: selected drugs to be used
	:upper for compatibility
	:return list of pvalues
	"""
	if ctrp_drugs is not None:
		netphix_ctrp_df = modules_df[modules_df["drug"].isin(ctrp_drugs)].copy()

	# call cross validation
	netphix_ctrp_df.fillna("", inplace=True)
	dec_muts = netphix_ctrp_df.dec.values
	inc_muts = netphix_ctrp_df.inc.values
	ctrp = netphix_ctrp_df.drug.values

	pvs = []
	num_rows = len(ctrp)
	for i in range(num_rows):
		drug = ctrp[i]
		if drug.endswith("(1)") or drug.endswith("(2)") or drug.endswith("(-)"):
			drug = drug[:-4]
		# ctrp_auc = auc_df.loc[drug]
		pvs.append(cross_val_ind(ctrp_auc_df, drug, ctrp_alt_df, dec_muts[i].split(","), inc_muts[i].split(","), method))
	netphix_ctrp_df["cv_"+method+"_pv"] = pvs

	return netphix_ctrp_df


def sig_df(df, label="cv_mut_pv", th=0.05):
    return df[df[label] < th]


# count number of significant modules
def count_sig(df, label, pv_th):
	"""
	:param df:
	:param label:
	:param pv_th:
	:return:
	"""
	sig_modules = sig_df(df, label, pv_th)
	num_sig_modules = sig_modules.shape[0]
	num_sig_drugs = len(set(sig_modules.drug.values))
	return num_sig_modules, num_sig_drugs


def count_sig_all(selected_modules, netphix_ctrp_df, pv_th=0.05, ctrp_pv_th=0.05):
	"""

	:param selected_modules:
	:param pv_th:
	:param netphix_ctrp_df:
	:param ctrp_pv_th:
	:return:
	"""
	sig_modules, sig_drugs = count_sig(selected_modules, "pv", pv_th)
	ctrp_modules, ctrp_drugs = count_sig(netphix_ctrp_df, "pv", pv_th)
	ctrp_sig_modules, ctrp_sig_drugs = count_sig(sig_df(netphix_ctrp_df, "pv", pv_th), "cv_mut_pv", ctrp_pv_th)
	module_ratio = ctrp_sig_modules/float(ctrp_modules)
	drug_ratio = ctrp_sig_drugs/float(ctrp_drugs)
	module_stats_dic = dict([("n_sig_modules", sig_modules),
							("n_tested_modules", ctrp_modules),
							("n_ctrp_sig_modules", ctrp_sig_modules),
							("ratio_confirmed_modules", module_ratio)])
	drug_stats_dic = dict([("n_sig_drugs", sig_drugs),
							 ("n_tested_drugs", ctrp_drugs),
							 ("n_ctrp_sig_drugs", ctrp_sig_drugs),
						   ("ratio_confirmed_drugs", drug_ratio)])

	return module_stats_dic, drug_stats_dic


