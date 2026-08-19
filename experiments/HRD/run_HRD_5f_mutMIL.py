#%%
"""HRD attMIL - SNV/indel attention MIL. 5-fold CV."""

import os
import random
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
import numpy as np
np.random.seed(SEED)
random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
from mutationMIL.Sample_MIL import InstanceModels, RaggedModels
from mutationMIL.KerasLayers import Losses, Metrics
from mutationMIL import DatasetsUtils
import pandas as pd
import pickle

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

cwd = "..."
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "weights", "HRD")
dropout = .4
model_name = 'DL_HRD_attMIL'
all_predictions = []

D, samples, sample_df = pickle.load(open(cwd + '/controlled_filters_combined_HRD_data_finished_20_pos.pkl', 'rb'))

strand_emb_mat = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D['strand_emb'] = strand_emb_mat[D['strand']]
chr_emb_mat = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D['chr_emb'] = chr_emb_mat[D['chr']]
frame_emb_mat = np.concatenate([np.zeros(3)[np.newaxis, :], np.diag(np.ones(3))], axis=0)
D['cds_emb'] = frame_emb_mat[D['cds']]

indexes = [np.where(D['sample_idx'] == idx) for idx in range(sample_df.shape[0])]
five_p = np.array([D['seq_5p'][i] for i in indexes], dtype='object')
three_p = np.array([D['seq_3p'][i] for i in indexes], dtype='object')
ref = np.array([D['seq_ref'][i] for i in indexes], dtype='object')
alt = np.array([D['seq_alt'][i] for i in indexes], dtype='object')
strand = np.array([D['strand_emb'][i] for i in indexes], dtype='object')

index_loader = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes for j in i], dropout=dropout)
five_p_loader = DatasetsUtils.Map.FromNumpyandIndices(five_p, tf.int16)
three_p_loader = DatasetsUtils.Map.FromNumpyandIndices(three_p, tf.int16)
ref_loader = DatasetsUtils.Map.FromNumpyandIndices(ref, tf.int16)
alt_loader = DatasetsUtils.Map.FromNumpyandIndices(alt, tf.int16)
strand_loader = DatasetsUtils.Map.FromNumpyandIndices(strand, tf.float32)

five_p_loader_eval = DatasetsUtils.Map.FromNumpy(five_p, tf.int16)
three_p_loader_eval = DatasetsUtils.Map.FromNumpy(three_p, tf.int16)
ref_loader_eval = DatasetsUtils.Map.FromNumpy(ref, tf.int16)
alt_loader_eval = DatasetsUtils.Map.FromNumpy(alt, tf.int16)
strand_loader_eval = DatasetsUtils.Map.FromNumpy(strand, tf.float32)

y_label = samples['class'][:, 0][:, np.newaxis]
y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)

D_cptac, samples_cptac, sample_df_cptac = pickle.load(open(cwd + '/HRD_cptac_data_finished_20_pos.pkl', 'rb'))
D_cptac['strand_emb'] = strand_emb_mat[D_cptac['strand']]
D_cptac['chr_emb'] = chr_emb_mat[D_cptac['chr']]
indexes_cptac = [np.where(D_cptac['sample_idx'] == idx) for idx in range(len(sample_df_cptac))]
five_p_cptac = np.array([D_cptac['seq_5p'][i] for i in indexes_cptac], dtype='object')
three_p_cptac = np.array([D_cptac['seq_3p'][i] for i in indexes_cptac], dtype='object')
ref_cptac = np.array([D_cptac['seq_ref'][i] for i in indexes_cptac], dtype='object')
alt_cptac = np.array([D_cptac['seq_alt'][i] for i in indexes_cptac], dtype='object')
strand_cptac = np.array([D_cptac['strand_emb'][i] for i in indexes_cptac], dtype='object')

five_p_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(five_p_cptac, tf.int16)
three_p_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(three_p_cptac, tf.int16)
ref_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(ref_cptac, tf.int16)
alt_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(alt_cptac, tf.int16)
strand_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(strand_cptac, tf.float32)

sample_df_cptac['HRD_binary'] = (sample_df_cptac['HRD_binary_paper'] == 'HRDpositive').astype(int)
y_label_cptac = sample_df_cptac['HRD_binary'].values[:, np.newaxis]
cptac_patient_ids = sample_df_cptac['PATIENT'].tolist()
print(f"CPTAC samples loaded: {len(sample_df_cptac)}")
print(f"  HRD-positive: {sample_df_cptac['HRD_binary'].sum()}")
print(f"  HRD-negative: {(1 - sample_df_cptac['HRD_binary']).sum()}")

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.002, patience=80, mode='max', restore_best_weights=True)]
losses = [Losses.BinaryCrossEntropy(from_logits=True)]
samples_list = sample_df['bcr_patient_barcode'].tolist()

for j in range(5):
    print(f"\n{'='*60}")
    print(f"Fold {j+1}/5")
    print(f"{'='*60}")

    df = pd.read_csv(cwd + f"/HRDfolds{j+1}.csv")
    df = df.rename(columns={"Patient ID": 'bcr_patient_barcode'})
    df = df.sort_values(by='bcr_patient_barcode')
    df = df[df['bcr_patient_barcode'].isin(samples_list)]
    df = df.drop_duplicates()
    index_mapping = sample_df[['bcr_patient_barcode']].reset_index()
    small_df_filtered = df[df['bcr_patient_barcode'].isin(index_mapping['bcr_patient_barcode'])]
    small_df_ordered = index_mapping.merge(small_df_filtered, on='bcr_patient_barcode', how='inner')
    small_df_ordered = small_df_ordered.set_index('index').sort_index()
    df = small_df_ordered.drop_duplicates()
    mapping = {"HRD_positive": 1, "HRD_negative": 0}
    df['labels_int'] = df['HRD_status'].map(mapping)
    train_df = df[df["split"] == "train"]
    train_msih = train_df[train_df["HRD_status"] == "HRD_positive"]
    train_nonmsih = train_df[train_df["HRD_status"] == "HRD_negative"]
    train_resampled_nonmsih = train_nonmsih.sample(n=len(train_msih), random_state=SEED)
    train_df = pd.concat([train_msih, train_resampled_nonmsih], axis=0).sample(frac=1, random_state=SEED)
    train = train_df.index.tolist()
    test_df = df[df["split"] == "test"].sample(frac=1, random_state=SEED)
    test = test_df.index.tolist()
    val_df = df[df["split"] == "validation"]
    val_msih = val_df[val_df["HRD_status"] == "HRD_positive"]
    val_nonmsih = val_df[val_df["HRD_status"] == "HRD_negative"]
    val_resampled_nonmsih = val_nonmsih.sample(n=len(val_msih), random_state=SEED)
    val_df = pd.concat([val_msih, val_resampled_nonmsih], axis=0).sample(frac=1, random_state=SEED)
    val = val_df.index.tolist()

    ds_train = tf.data.Dataset.from_tensor_slices(train)
    ds_train = ds_train.apply(DatasetsUtils.Apply.SubSample(batch_size=128, ds_size=len(train)))
    ds_train = ds_train.map(lambda x: ((index_loader(x),),))
    ds_train = ds_train.map(lambda x: ((five_p_loader(x[0], x[1]),
                                        three_p_loader(x[0], x[1]),
                                        ref_loader(x[0], x[1]),
                                        alt_loader(x[0], x[1]),
                                        strand_loader(x[0], x[1]),
                                        ),
                                       y_label_loader(x[0]),
                                       ))
    ds_train = ds_train.prefetch(1)
    ds_valid = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(val),
                                                    three_p_loader_eval(val),
                                                    ref_loader_eval(val),
                                                    alt_loader_eval(val),
                                                    strand_loader_eval(val),
                                                    ),
                                                   tf.gather(y_label, val),
                                                   ))
    ds_valid = ds_valid.batch(len(val), drop_remainder=False)

    ds_test = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(test),
                                                   three_p_loader_eval(test),
                                                   ref_loader_eval(test),
                                                   alt_loader_eval(test),
                                                   strand_loader_eval(test),
                                                   ),
                                                  tf.gather(y_label, test),
                                                  ))
    ds_test = ds_test.batch(len(test), drop_remainder=False)

    val_cptac = list(range(len(sample_df_cptac)))
    ds_test_cptac = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval_cptac(val_cptac),
                                                         three_p_loader_eval_cptac(val_cptac),
                                                         ref_loader_eval_cptac(val_cptac),
                                                         alt_loader_eval_cptac(val_cptac),
                                                         strand_loader_eval_cptac(val_cptac),
                                                         ),
                                                        tf.gather(y_label_cptac, val_cptac),
                                                        ))
    ds_test_cptac = ds_test_cptac.batch(len(val_cptac), drop_remainder=False)

    sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=128)
    mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=y_label.shape[-1], mil_hidden=(256, 128), attention_layers=[], dropout=.6, instance_dropout=.6, regularization=.1, input_dropout=dropout)
    mil.model.compile(loss=losses,
                      metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    mil.model.fit(ds_train,
                  steps_per_epoch=10,
                  validation_data=ds_valid,
                  epochs=500,
                  callbacks=callbacks,
                  )

    weights = mil.model.get_weights()
    with open(os.path.join(WEIGHTS_DIR, f'HRD_attMIL_weights_fold{j+1}.pkl'), 'wb') as f:
        pickle.dump(weights, f)
    print(f"Saved weights: {os.path.join(WEIGHTS_DIR, f'HRD_attMIL_weights_fold{j+1}.pkl')}")

    y_pred = tf.sigmoid(mil.model.predict(ds_test)).numpy()
    for idx, (score, label) in enumerate(zip(y_pred.flatten(), y_label[test].flatten())):
        all_predictions.append({
            'fold': j,
            'patient_id': test_df['bcr_patient_barcode'].iloc[idx],
            'cohort': 'TCGA_test',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })
    print(f"  TCGA test: {len(y_pred)} predictions")

    y_pred_cptac = tf.sigmoid(mil.model.predict(ds_test_cptac)).numpy()
    for idx, (score, label) in enumerate(zip(y_pred_cptac.flatten(), y_label_cptac[val_cptac].flatten())):
        all_predictions.append({
            'fold': j,
            'patient_id': cptac_patient_ids[idx],
            'cohort': 'CPTAC_full',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })
    print(f"  CPTAC full (scarHRD): {len(y_pred_cptac)} predictions")

print(f"\n{'='*60}")
print("Saving predictions...")
print(f"{'='*60}")

predictions_df = pd.DataFrame(all_predictions)
output_dir = "predictions_DL_attMIL/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions_HRD_attMIL_all_folds.csv")
predictions_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Total predictions: {len(predictions_df)}")
print(f"Cohorts: {predictions_df['cohort'].unique().tolist()}")

for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    cohort_path = os.path.join(output_dir, f"predictions_HRD_attMIL_{cohort}.csv")
    cohort_df.to_csv(cohort_path, index=False)
    print(f"  {cohort}: {len(cohort_df)} samples")

print(f"\n{'='*60}")
print("Prediction Summary")
print(f"{'='*60}")
for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    n_pos = cohort_df['label'].sum()
    n_neg = len(cohort_df) - n_pos
    print(f"  {cohort}: {len(cohort_df)} samples (HRD+: {n_pos}, HRD-: {n_neg})")

print(f"\n{'='*60}")
print("Next Step: Run evaluation pipeline")
print(f"{'='*60}")
print(f"python run_evaluation_pipeline.py \\")
print(f"  --input {output_path} \\")
print(f"  --output-dir evaluation_results_DL_attMIL/")

# %%
