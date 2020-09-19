# NETPHIX

NETPHIX (NETwork-to-PHenotype association LeveragIng eXlusivity) is a computational tool
to identify mutated subnetworks that are associated with a continuous phenotype. 
Using an integer linear program with properties for mutual exclusivity and interactions among genes,
NETPHIX finds an optimal set of genes maximizing the association.
For more details, see [1].

### Requirements

* Linux/Mac OS/Windows
* python version 3.6.2/pandas version 0.24.2
* CPLEX version 12.9.0.0 (set PYTHONPATH to the cplex location);
    - install the CPLEX-Python modules usig the script "setup.py" located in "yourCplexhome/python/"

            $ python setup.py install
    -  install python CPLEX interface

            $ pip install cplex

### How to use

#### run_NETPHIX.py

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


Examples: 

Run NETPHIX with drug target 1
(combined, k=3, mutation frequency between 1 and 0.01, network density 0.5, normalization method=z, with penalty)

    $ python run_NETPHIX.py data/AlterationsV2_final.txt.gz  data/gdsc_auc/Target1.txt combined 3 1 0.01 0.5 z p data/HumanStringNet.txt results/test/result_combined_3_p.txt


Run NETPHIX with drug target 1 (for separate model)
(separate model (combined2), k=4, mutation frequency between 1 and 0.01, network density 0.5, normalization method=z, with penalty)

    $ python run_NETPHIX.py data/AlterationsV2_final.txt.gz  data/gdsc_auc/Target1.txt combined2 4 1 0.01 0.5 z p data/HumanStringNet.txt results/test/result_sep_4_p.txt


Run NETPHIX with drug target 1 (for combined model)
(combined, k=3, mutation frequency between 1 and 0.01, network density 0.5, normalization method=z, no penalty)

    $ python run_NETPHIX.py data/AlterationsV2_final.txt.gz  data/gdsc_auc/Target1.txt combined 3 1 0.01 0.5 z np data/HumanStringNet.txt results/test/result_combined_3_np.txt

Run NETPHIX with drug target 1 and compute p-value with target permutation (assuming target permutation is already performed). See below how to run permutation test
(combined, k=3, mutation frequency between 0.25 and 0.01, network density 0.5, normalization method=z)

    $ python run_NETPHIX.py data/AlterationsV2_final.txt.gz data/gdsc_auc/Target1.txt combined 3 1 0.01 0.5 z p data/HumanStringNet.txt results/test/result_combined_3_p_with_pv.txt -tp results/test/perm_target_combined_3.txt




### target_permute_NETPHIX.py

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

Examples:
   
    
Run target permutation 10 times with drug target 1 (use combined model for correlation)
(k=3, mutation frequency between 0.25 and 0.01, network density 0.5, normalization method=z, with penalty)

    $ python target_permute_NETPHIX.py data/AlterationsV2_final.txt.gz data/gdsc_auc/Target1.txt combined 3 0.25 0.01 0.5 z p data/HumanStringNet.txt  10  results/test/perm_target_combined_3.txt

## Identifying Drug Sensitivity network using NETPHIX

### 1. Run Netphix 
    run_NETPHIX.py
    Input:
        AlterationsV2_final.txt 
        target datafiles are at data/gdsc_auc
        data/HumanStringNet.txt
    Output:
    
### 2. Permutation test
    target_permute_NETPHIX.py
    Input:
        AlterationsV2_final.txt 
        target datafiles are at data/gdsc_auc
        data/HumanStringNet.txt
    
### 3. Selecting Final modules 
    select_final_modules.ipynb
    Input:
        netphix files with pv in results/gdsc/   
        example file for Drug 1          
    Output:
        max_sig_combined_modules_0.05.tsv
    

### Data and  result files

Source files

    NETPHIX core files
        run_NETPHIX.py
            - run NETPHIX
        target_permute_NETPHIX.py
            - run permutation test
        select_final_modules.ipynb
            - summarize NETPHIX results and pick the final modules

    Utility functions
        cv_utils.py
        dist_utils.py
        gen_simulated.py
        netphix_utils.py
        permute_utils.py
        post_analysis.py
        select_sig_modules.py
       
    Evalutions     
        regress_nephix.ipynb 
        anova_netphix_uncover.ipynb
        comp_dist_target_all.ipynb
        

Data directory 

    data/
        HumanStringNet.txt: interaction network
        AlterationsV2_final.txt.gz : gene alteration file
        gdsc_auc/Target*.txt : drug response auc files
        
        gdsc_drug_targets.tsv: GDSC drug targets
        drug_target_id.txt:
        ctrp_auc_processed.txt: CTRP AUC data 
        merged_uncover_modules_0.05.tsv: UNCOVER modules
         

Results directory
    
    results/
        test/
            result_combined_3_p.txt
            result_sep_4_p.txt
            result_combined_3_np.txt
            perm_target_combined_3.txt
            result_combined_3_p_with_pv.txt
        gdsc/ # netphix candidate modules
            depmap_results_$drug_id$_combined_$k$.txt
            depmap_results_$drug_id$_combined2_$k$.txt
        # netphix final modules
        max_sig_combined_modules_0.05.tsv
        max_sig_combined_modules_ctrp_cv_0.05.tsv 
        combination_candidates.tsv
                

## References

[1] Identifying Drug Sensitivity Subnetworks with NETPHIX. https://www.biorxiv.org/content/10.1101/543876v1?rss=1


## Authors

* **Yoo-Ah Kim** (kimy3@ncbi.nlm.nih.gov) 




