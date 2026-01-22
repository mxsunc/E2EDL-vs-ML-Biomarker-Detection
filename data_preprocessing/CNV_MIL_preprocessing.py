import pandas as pd
import numpy as np
import pickle
import pyranges as pr

# load chromosomal information
genes_df = pd.read_csv(".../genes_grch38.bed", sep="\t")
genes_df.columns = ["Chromosome", "Start", "End", "Gene", "GeneType"]
genes_df = genes_df[genes_df["GeneType"] == "protein_coding"]
genes_df = genes_df[~genes_df["Gene"].str.startswith("MT-")]
genes_df = genes_df[~genes_df["Gene"].isin(FoundationOne324_list)]

genes_df["Chromosome"] = genes_df["Chromosome"].str.replace("chr", "", regex=False)
genes_df["Chromosome"] = genes_df["Chromosome"].replace({"X": "22", "Y": "23"})

genes_pr = pr.PyRanges(genes_df[["Chromosome", "Start", "End", "Gene"]])

# load segment data of all patients
seg_path = ".../combined_study_segments.seg"
seg = pd.read_csv(seg_path, sep="\t", comment="#", dtype=str)

seg["chrom"] = seg["chrom"].astype(str).str.replace("chr", "", regex=False)

# check if one segment row is inside any gene
def seg_overlaps_gene(row):
    chrom = str(row["chrom"])
    start = int(row["loc.start"])
    end   = int(row["loc.end"])
    try:
        g = genes_pr[chrom].df
    except KeyError:
        return False

    if g.empty:
        return False
    hit = (
        ((g["Start"] <= start) & (g["End"] >= start)) |
        ((g["Start"] <= end)   & (g["End"] >= end))
    )
    return bool(hit.any())

mask = seg.apply(seg_overlaps_gene, axis=1)
seg = seg[mask].copy()

# instance features
seg["loc.start"] = seg["loc.start"].astype(int)
seg["loc.end"]   = seg["loc.end"].astype(int)
seg["num.mark"]  = seg["num.mark"].astype(float)
seg["seg.mean"]  = seg["seg.mean"].astype(float)
seg["length_bp"] = seg["loc.end"] - seg["loc.start"] + 1
seg["log_length"] = np.log10(seg["length_bp"].clip(lower=1))
seg["cn"] = 2.0 * (2.0 ** seg["seg.mean"])
seg["center_bp"] = 0.5 * (seg["loc.start"] + seg["loc.end"])
seg["chr_len"] = seg.groupby("chrom")["loc.end"].transform("max")
seg["center_rel_chr"] = seg["center_bp"] / seg["chr_len"].replace(0, np.nan)
seg["is_gain"] = (seg["seg.mean"] > 0.2).astype(int)
seg["is_loss"] = (seg["seg.mean"] < -0.2).astype(int)
seg["log_num_mark"] = np.log10(seg["num.mark"] + 1.0)

chrom_to_idx = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
    "6": 5, "7": 6, "8": 7, "9": 8, "10": 9,
    "11": 10, "12": 11, "13": 12, "14": 13, "15": 14,
    "16": 15, "17": 16, "18": 17, "19": 18, "20": 19,
    "21": 20, "22": 21, "X": 22, "Y": 23,
}
seg["chrom_idx"] = seg["chrom"].astype(str).map(chrom_to_idx)

feature_cols = [
    "seg.mean",
    "log_length",
    "center_rel_chr",
    "log_num_mark",
    "cn",
    "is_gain",
    "is_loss",
    "chrom_idx",
]

# create MIL bags
bags = {}
for sid, df_id in seg.groupby("ID"):
    x_cont = df_id[feature_cols].values.astype(np.float32)
    bags[sid] = {
        "x_cont": x_cont
    }

with open(".../tcga_cnv_mil_bags.pkl", "wb") as f:
    pickle.dump(bags, f)