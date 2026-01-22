#%%
import os
import shutil
import pandas as pd
import pyranges as pr
#%%
# generate lookup table for genes
gtf_file = ".../gencode.v38.annotation.gtf" # gencode v38 genome annotations

genes = []
with open(gtf_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if fields[2] != "gene":
            continue
        chrom = fields[0]
        start = int(fields[3]) - 1
        end = int(fields[4])
        info = fields[8]

        gene_name = None
        gene_type = None

        for entry in info.split(";"):
            entry = entry.strip()
            if entry.startswith("gene_name"):
                gene_name = entry.split('"')[1]
            elif entry.startswith("gene_type") or entry.startswith("gene_biotype"):
                gene_type = entry.split('"')[1]

        if gene_name:
            genes.append([chrom, start, end, gene_name, gene_type])

genes_df = pd.DataFrame(genes, columns=["Chromosome", "Start", "End", "Gene", "GeneType"])

# sort out genes
genes_df = genes_df[genes_df["GeneType"] == "protein_coding"]
genes_df = genes_df[~genes_df["Gene"].str.startswith("MT-")]

# convert to pyranges
genes_pr = pr.PyRanges(genes_df)
#%%
# load metadata
base_dir = ".../gdc_sample_sheet.tsv" # GDC samples

sample_sheet = pd.read_csv(base_dir, sep="\t")

required_cols = ['File ID', 'File Name', 'Sample ID', 'Case ID']

selected_rows = []
for patient_id, group in sample_sheet.groupby("patient_id"):
        selected_rows.append(group.iloc[0])

final_df = pd.DataFrame(selected_rows).reset_index(drop=True)
#%%
# load segments
seg_folder = "..." # TCGA CNV segment files 

output = {}
for file in os.listdir(seg_folder):
    if file.endswith(".seg.txt"):
        patient_id = final_df.loc[final_df["File Name"] == file, "patient_id"].values[0]
        seg_path = os.path.join(seg_folder, file)
        seg_df = pd.read_csv(seg_path, sep="\t")

        seg_pr = pr.PyRanges(seg_df[["Chromosome", "Start", "End", "Segment_Mean"]].copy())

        joined = seg_pr.join(genes_pr) #Intersect segments with genes
        if joined.df.empty:
            continue
 
        gene_cnv = joined.df.groupby("Gene")["Segment_Mean"].mean().to_dict() #Collapse to average CNV per gene
        output[patient_id] = gene_cnv

matrix_df = pd.DataFrame(output).fillna(0).T
matrix_df.index.name = "patient_id"
matrix_df.to_csv(".../cnv_seg_matrix.tsv", sep="\t")