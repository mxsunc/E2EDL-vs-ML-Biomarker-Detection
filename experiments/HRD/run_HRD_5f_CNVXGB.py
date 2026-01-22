#%%
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, balanced_accuracy_score,f1_score
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import CalibratedClassifierCV
from experiments.helpers.align_dfs import align_dfB_to_dfA
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

def asfloat32(x):
    return np.asarray(x, dtype=np.float32)

best_params = {
    "n_estimators": 600,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "learning_rate": 0.01,
    "max_depth": None,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    }

gene_pos_df = pd.read_csv("/mnt/bulk-uranus/michaela/ATGC/ATGC/CNV/helpers/CNV_pos_embeddings_100k.csv")

chrom_order = {**{str(i): i for i in range(1,23)}, "X": 23, "Y": 24, "MT": 25}
genes_ordered = gene_pos_df["gene"].tolist()

input_path = ".../tcga_cnv_seg_matrix.tsv"
all_cancers_df = pd.read_csv(input_path, sep="\t")
all_cancers_df = all_cancers_df.set_index("patient_id")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()
all_cancers_df["bcr_patient_barcode"] = all_cancers_df.index

gene_pos_df = pd.read_csv(".../CNV_pos_embeddings_100k.csv")

chrom_order = {**{str(i): i for i in range(1,23)}, "X": 23, "Y": 24, "MT": 25}
genes_ordered = gene_pos_df["gene"].tolist()

input_path = ".../cptac_cnv_seg_matrix.tsv"
all_cancers_df_cptac = pd.read_csv(input_path, sep="\t")

df_hrd_cptac = pd.read_csv(".../cptac_HRDstatus.csv")
df_hrd_cptac = df_hrd_cptac[df_hrd_cptac["tissue"].isin(["Pancreas","Bronchus and lung","Uterus, NOS"])]
df_hrd_cptac = df_hrd_cptac.dropna(subset=["HRD_Binary_us"]).copy()
df_hrd_cptac["HRD_Binary_us"] = df_hrd_cptac["HRD_Binary_us"].map({"WT": 0, "MUT": 1})

df_merged_cptac = all_cancers_df_cptac.merge(df_hrd_cptac, on="PATIENT", how="inner")
X_cptac_df, y_cptac = df_merged_cptac.iloc[:, 1:-10], df_merged_cptac['HRD_Binary_us']
X_cptac = asfloat32(X_cptac_df.values)
y_cptac = np.asarray(y_cptac).ravel().astype(np.int32)

for j in range(5):
    df = pd.read_csv(f".../HRDfolds{j+1}_withlungucec.csv")
    df = df.drop(columns={"BCR"})
    df = df.rename(columns={"Patient ID":"bcr_patient_barcode"})
    df = df.dropna(subset=["HRD_status"])
    df["HRD_status"] = df["HRD_status"].map({"HRD_negative": 0, "HRD_positive": 1})
    df_merged_together = df.merge(all_cancers_df, on="bcr_patient_barcode", how="right")
    train_df = df_merged_together[df_merged_together["split"]=="train"]
    train_msih = train_df[train_df["HRD_status"]==1]
    train_nonmsih  = train_df[train_df["HRD_status"]==0]
    train_resampled_nonmsih = train_nonmsih.sample(n=len(train_msih))
    train_df = pd.concat([train_msih, train_resampled_nonmsih], axis=0)
    train_data = train_df.sample(frac=1)
    val_df = df_merged_together[df_merged_together["split"]=="validation"]
    val_msih = val_df[val_df["HRD_status"]==1]
    val_nonmsih  = val_df[val_df["HRD_status"]==0]
    val_resampled_nonmsih = val_nonmsih.sample(n=len(val_msih))
    val_df = pd.concat([val_msih, val_resampled_nonmsih], axis=0)
    val_data = val_df.sample(frac=1)
    test_df = df_merged_together[df_merged_together["split"]=="test"]
    test_data = test_df.sample(frac=1)
    X_train = train_data.iloc[:, 14:-1]
    y_train = train_data[train_data["split"]=="train"]["HRD_status"].astype(int).values

    shared = [g for g in genes_ordered if g in X_train.columns]
    X_train = X_train.loc[:, shared].copy()
    X_val = val_data.iloc[:, 14:-1]
    y_val = val_data[val_data["split"]=="validation"]["HRD_status"].astype(int).values
    X_val = X_val.loc[:, shared].copy()

    X_test = test_data.iloc[:, 14:-1]
    y_test = test_data[test_data["split"]=="test"]["HRD_status"].astype(int).values
    X_test = X_test.loc[:, shared].copy()

    y_train = np.asarray(y_train).ravel().astype(np.int32)
    y_val = np.asarray(y_val).ravel().astype(np.int32)
    y_test = np.asarray(y_test).ravel().astype(np.int32)

    best_params = {
    "n_estimators": 600,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "learning_rate": 0.01,
    "max_depth": None,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    }
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    youden_j = tpr - fpr
    best_threshold_index = np.argmax(youden_j)
    best_threshold = thresholds[best_threshold_index]
    y_pred_binary = (y_pred >= best_threshold).astype(int)
    
    y_pred_cptac = model.predict_proba(X_cptac)[:,1]
    y_pred_binary_cptac = (y_pred_cptac >= best_threshold).astype(int)
