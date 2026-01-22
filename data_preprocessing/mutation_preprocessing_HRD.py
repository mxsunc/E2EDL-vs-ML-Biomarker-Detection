#%%
import pickle
import pandas as pd
import numpy as np
import re

path = '...'
tcga_maf = pickle.load(open(path+'controlled_allmut_combined_HRD_tcga_maf_table_20.pkl', 'rb'))
samples = pickle.load(open(path+ 'controlled_allmut_combined_HRD_sample_table.pkl', 'rb'))
panels= pickle.load(open(path+ 'controlled_allmut_combined_HRD_tcga_panel_table_20.pkl', 'rb'))

samples = samples.drop_duplicates()
samples = samples.loc[~pd.isna(samples['HRD_status'])]
samples.reset_index(inplace=True, drop=True)
tcga_maf = tcga_maf.loc[tcga_maf['Tumor_Sample_Barcode'].isin(samples.Tumor_Sample_Barcode.values)]
tcga_maf.dropna(inplace=True, subset=['Ref'])
tcga_maf.reset_index(inplace=True, drop=True)
tcga_maf = pd.merge(tcga_maf, samples.Tumor_Sample_Barcode.reset_index(), how='left', on='Tumor_Sample_Barcode')

#get index
samples_idx = tcga_maf['index'].values

# 5p, 3p, ref, alt
nucleotide_mapping = {'-': 0, 'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
seqs_5p = np.stack(tcga_maf.five_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_3p = np.stack(tcga_maf.three_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_ref = np.stack(tcga_maf.Ref.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
seqs_alt = np.stack(tcga_maf.Alt.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)

# chr, pos
chromosome_mapping = dict(zip([str(i) for i in list(range(1, 23))] + ['X', 'Y'], list(range(1, 25))))
gen_chr = np.array([chromosome_mapping[i] for i in tcga_maf.Chromosome.values])
chromosome_sizes = pd.read_csv(path + 'chromInfo.hg19.tsv', sep='\t', header=None)
chromosome_sizes.columns = chromosome_sizes.iloc[0]
chromosome_sizes = chromosome_sizes.drop(chromosome_sizes.index[0])
chromosome_sizes= chromosome_sizes[chromosome_sizes.index < 25] 
chrom_names = chromosome_sizes.hg19g0.tolist()
new_names = [x[3:] for x in chrom_names]
chromosome_sizes["hg19g0"] = new_names
chromosome_sizes['size'] = chromosome_sizes['size'].astype(int)
chromosome_sizes = dict(zip(chromosome_sizes['hg19g0'], chromosome_sizes['size']))
desired_order = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
sorted_chromosomes = sorted(chromosome_sizes.keys(), key=lambda x: desired_order.index(x))
cumulative_sizes = {}
total_genome_size = 0
for chrom in sorted_chromosomes:
    cumulative_sizes[chrom] = total_genome_size
    total_genome_size += chromosome_sizes[chrom]

#positions of mutations
def calculate_genome_wide_position(row):
    chrom = row['Chromosome']
    pos = row['Start_Position']
    return cumulative_sizes[chrom] + pos

gen_pos = tcga_maf.apply(calculate_genome_wide_position, axis=1)
gen_pos = gen_pos / total_genome_size
gen_pos = np.array(gen_pos.tolist())

cds = tcga_maf['CDS_position'].astype(str).apply(lambda x: (int(x) % 3) + 1 if re.match('^[0-9]+$', x) else 0).values

instances = {'sample_idx': samples_idx,
             'seq_5p': seqs_5p,
             'seq_3p': seqs_3p,
             'seq_ref': seqs_ref,
             'seq_alt': seqs_alt,
             'chr': gen_chr,
             'pos_float': gen_pos,
             'cds': cds}

A = samples.HRD_status.astype('category')
classes = A.cat.categories.values
classes_onehot = np.eye(len(classes))[A.cat.codes]

samples_dict = {'cancer': samples.type.values,
           'histology': samples.histological_type.values,
           'age': samples.age_at_initial_pathologic_diagnosis.values,
           'class': classes_onehot,
           'classes': classes,
           'bcr_patient_barcode': samples.bcr_patient_barcode }

variant_encoding = np.array([0, 2, 1, 4, 3])
instances['seq_5p'] = np.stack([instances['seq_5p'], variant_encoding[instances['seq_3p'][:, ::-1]]], axis=2)
instances['seq_3p'] = np.stack([instances['seq_3p'], variant_encoding[instances['seq_5p'][:, :, 0][:, ::-1]]], axis=2)
t = instances['seq_ref'].copy()
i = t != 0
t[i] = variant_encoding[instances['seq_ref'][:, ::-1]][i[:, ::-1]]
instances['seq_ref'] = np.stack([instances['seq_ref'], t], axis=2)
t = instances['seq_alt'].copy()
i = t != 0
t[i] = variant_encoding[instances['seq_alt'][:, ::-1]][i[:, ::-1]]
instances['seq_alt'] = np.stack([instances['seq_alt'], t], axis=2)
del i, t

instances['strand'] = tcga_maf['STRAND'].astype(str).apply(lambda x: {'.': 0, '-1': 1, '1': 2}[x]).values

with open(path +'controlled_allmut_combined_HRD_data_finished_20_pos.pkl', 'wb') as f:
    pickle.dump([instances, samples_dict, samples], f)
# %%
