#%%
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from matplotlib.colors import LinearSegmentedColormap
import xgboost as xgb
from builtins import min
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import LinearSegmentedColormap
from numpy import mean, std
import shap

cwd = "..." 
tcga_df = pd.read_csv(cwd + '/tcga_sbs_mutation_catalogues_norm.csv')
tcga_df = tcga_df.rename(columns={"Patient_ID":"bcr_patient_barcode"})
tcga_df_id = pd.read_csv(cwd + '/tcga_id_mutation_catalogues_norm.csv')
tcga_df_id = tcga_df_id.rename(columns={"Patient_ID":"bcr_patient_barcode"})
merged_df = pd.merge(tcga_df, tcga_df_id, on='bcr_patient_barcode', how='inner')
tcga_df = merged_df[[col for col in merged_df.columns if col != 'bcr_patient_barcode'] + ['bcr_patient_barcode']]

precisions = []
recalls = []
tprs = []
fprs = []
auc_scores = []
auc_scores_pr = []
num_folds = 5
for i in range(1):
  for j in range(num_folds):
      df = pd.read_csv(cwd+"/HRDfolds"+str(j+1)+".csv")
      df =  df.rename(columns={"Patient ID":"bcr_patient_barcode"})
      df_merged = df.merge(tcga_df, on="bcr_patient_barcode", how="inner")
      X = df_merged.iloc[:, -179:]
      label_mapping = {'HRD_positive': 1, 'HRD_negative': 0}
      df_merged['label'] = df_merged['HRD_binary_paper'].map(label_mapping)
      y = df_merged["label"]
      train_df = df_merged[df_merged["split"]=="train"]
      train_msih = train_df[train_df["HRD_binary_paper"]=="HRD_positive"]
      # apply undersampling for the majority class like for attMIL model
      train_nonmsih  = train_df[train_df["HRD_binary_paper"]=="HRD_negative"]
      train_resampled_nonmsih = train_nonmsih.sample(n=len(train_msih))
      train_df = pd.concat([train_msih, train_resampled_nonmsih], axis=0)
      train_data = train_df.sample(frac=1)
      test_df = df_merged[df_merged["split"]=="test"] 
      test_msih  = test_df[test_df["HRD_binary_paper"]=="HRD_positive"]
      test_nonmsih  = test_df[test_df["HRD_binary_paper"]=="HRD_negative"]
      test_resampled_nonmsih = test_nonmsih.sample(n=len(test_msih))
      test_df = pd.concat([test_msih, test_resampled_nonmsih], axis=0)
      test_data = test_df.sample(frac=1)
      val_df = df_merged[df_merged["split"]=="validation"]
      val_msih  = val_df[val_df["HRD_binary_paper"]=="HRD_positive"]
      val_nonmsih  = val_df[val_df["HRD_binary_paper"]=="HRD_negative"]
      val_resampled_nonmsih = val_nonmsih.sample(n=len(val_msih))
      val_df = pd.concat([val_msih, val_resampled_nonmsih], axis=0)
      val_data = val_df.sample(frac=1)
      X_train, y_train = train_data.iloc[:, -180:-1], train_data['label']
      X_test, y_test = test_data.iloc[:, -180:-1], test_data['label']
      X_val, y_val = val_data.iloc[:, -180:-1], val_data['label']

      # train model
      model = xgb.XGBClassifier()
      model.fit(X_train, y_train)

      #evaluate model
      y_pred = model.predict(X_test)
      accuracy = balanced_accuracy_score(y_test, y_pred)
      y_pred_proba = model.predict_proba(X_test)[:, 1]
      roc_auc = roc_auc_score(y_test, y_pred_proba)
      fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
      tprs.append(tpr)
      fprs.append(fpr)
      roc_auc = auc(fpr, tpr)
      auc_scores.append(roc_auc)
      f1 = f1_score(y_test, y_pred)
      precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
      precisions.append(precision)
      recalls.append(recall)
      auc_scores_pr.append(auc(recall, precision))
      explainer = shap.TreeExplainer(model)
      shap_values = explainer.shap_values(X_test)
      colors = ["#83c7e9","#185d85", "#bc1e2d", "#f4969d"]
      cmap_name = 'custom_coolwarm'
      custom_coolwarm = LinearSegmentedColormap.from_list(cmap_name, colors)
      shap.summary_plot(shap_values, X_test, cmap=custom_coolwarm)
      shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])

plt.figure(figsize=(4,4))
for j in range(num_folds):
  if j == 4:
    plt.plot(fprs[j], tprs[j], color='#e85f6a', lw=2, label=f'AUC = {mean(auc_scores):.3f} ± {std(auc_scores):.3f}')
  else:
    plt.plot(fprs[j], tprs[j], color='#e85f6a', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
plt.xlim([-0.005, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(4,4))
aus = np.array(auc_scores_pr)
mea = np.mean(aus)
std = np.std(aus)
min = 1.0
for j in range(5):
  pre = precisions[j]
  if min > pre[1]:
      min = pre[1]
  if j == 4:
    plt.plot(recalls[j], precisions[j], color='#e85f6a', lw=2, label=f'AUC = {mea:.3f} ± {std:.3f}')
  else:
    plt.plot(recalls[j], precisions[j], color='#e85f6a', lw=2) 
plt.plot([0, 1], [min-0.01, min-0.01], color='navy', lw=2, linestyle='--')
plt.xlabel('Recall', fontsize="12")
plt.ylabel('Precision', fontsize="12")
plt.ylim(0,1.01)
plt.xlim(-0.01,1.)
plt.title('Precision-Recall Curve',fontweight = 'bold')
plt.legend(loc="lower right", fontsize="10")
plt.show()