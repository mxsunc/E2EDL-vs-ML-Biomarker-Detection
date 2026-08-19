#%%
"""HRD Multimodal DL - SNV+CNV intermediate fusion (cross-attention, staged fine-tuning). 5-fold CV."""

import os
import sys
import math
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, MultiHeadAttention, Concatenate
from mutationMIL.Sample_MIL import InstanceModels, RaggedModels
from mutationMIL.KerasLayers import Metrics
from mutationMIL import DatasetsUtils

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[-1], True)
    tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')
    print(f"Using GPU: {physical_devices[-1]}")
else:
    print("No GPU available, running on CPU")

cwd = ".../"
cwd_cptac = ".../"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "weights", "HRD")
dropout = 0.0

D, samples, sample_df = pickle.load(open(cwd + 'controlled_filters_combined_HRDwithlungucec_data_finished_20_pos.pkl', 'rb'))

D_cptac, samples_cptac, sample_df_cptac = pickle.load(open(cwd_cptac + 'HRDwithlungucec_cptac_all_data_finished_20_pos_filter_only.pkl', 'rb'))
if "index" in sample_df_cptac.columns:
    sample_df_cptac = sample_df_cptac.drop(columns=["index"])
sample_df_cptac = sample_df_cptac.drop_duplicates(["Patient ID"])
sample_df_cptac['HRD_binary'] = (sample_df_cptac['HRD_binary_paper'] == 'HRDpositive').astype(int)

print(f"CPTAC samples loaded: {len(sample_df_cptac)}")
print(f"  HRD-positive: {sample_df_cptac['HRD_binary'].sum()}")
print(f"  HRD-negative: {(1 - sample_df_cptac['HRD_binary']).sum()}")

print("Loading TCGA CNV data...")
all_cancers_df = pd.read_csv(cwd + "CNV/all_cancers_cnv_matrix_int.csv", index_col="sample_barcode")
all_cancers_df["patient_id"] = all_cancers_df.index.str[:12]
all_cancers_df = all_cancers_df.set_index("patient_id")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()

samples_list_snv = sample_df['bcr_patient_barcode'].tolist()
samples_list_cnv = all_cancers_df.index.tolist()
common_samples = list(set(samples_list_snv) & set(samples_list_cnv))
print(f"  TCGA: {len(samples_list_snv)} SNV samples, {len(samples_list_cnv)} CNV samples, {len(common_samples)} in both")

sample_df = sample_df[sample_df['bcr_patient_barcode'].isin(common_samples)].copy()
all_cancers_df = all_cancers_df[all_cancers_df.index.isin(common_samples)]
all_cancers_df = all_cancers_df.sort_index()
all_cancers_df["bcr_patient_barcode"] = all_cancers_df.index
all_cancers_df = all_cancers_df.reset_index(drop=True)
print(f"  TCGA after inner join: {len(sample_df)} samples")

print("Loading CPTAC CNV data...")
all_cancers_df_cptac = pd.read_csv(cwd_cptac + "CNV/all_cancers_cnv_matrix_int.csv", index_col="sample_barcode")
all_cancers_df_cptac = all_cancers_df_cptac[~all_cancers_df_cptac.index.duplicated(keep="first")].copy()

samples_list_snv_cptac = sample_df_cptac['Patient ID'].tolist()
samples_list_cnv_cptac = all_cancers_df_cptac.index.tolist()
common_samples_cptac = list(set(samples_list_snv_cptac) & set(samples_list_cnv_cptac))
print(f"  CPTAC: {len(samples_list_snv_cptac)} SNV samples, {len(samples_list_cnv_cptac)} CNV samples, {len(common_samples_cptac)} in both")

all_cancers_df_cptac = all_cancers_df_cptac[all_cancers_df_cptac.index.isin(common_samples_cptac)]
all_cancers_df_cptac = all_cancers_df_cptac.sort_index()
sample_df_cptac = sample_df_cptac[sample_df_cptac['Patient ID'].isin(common_samples_cptac)].copy()
sample_df_cptac = sample_df_cptac.set_index('Patient ID').reindex(all_cancers_df_cptac.index).reset_index()
sample_df_cptac = sample_df_cptac.rename(columns={'index': 'sample_barcode'})
sample_df_cptac = sample_df_cptac.dropna(subset=['HRD_binary'])
sample_df_cptac = sample_df_cptac.drop_duplicates()
cptac_patient_ids = sample_df_cptac['sample_barcode'].tolist()
print(f"  CPTAC after inner join: {len(sample_df_cptac)} samples")

tcga_gene_cols = [c for c in all_cancers_df.columns if c != "bcr_patient_barcode"]
all_cancers_df = all_cancers_df[tcga_gene_cols]
all_cancers_df_cptac = all_cancers_df_cptac.reindex(columns=tcga_gene_cols, fill_value=0)
print(f"\nTCGA CNV matrix shape: {all_cancers_df.shape}")
print(f"CPTAC CNV matrix shape: {all_cancers_df_cptac.shape}")

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

strand_emb_mat_cptac = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D_cptac['strand_emb'] = strand_emb_mat_cptac[D_cptac['strand']]
chr_emb_mat_cptac = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D_cptac['chr_emb'] = chr_emb_mat_cptac[D_cptac['chr']]

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

y_label_cptac = sample_df_cptac['HRD_binary'].values[:, np.newaxis]

BATCH_SIZE = 64
N_CN_FEATURES = all_cancers_df.shape[1]
LATENT_DIM = 256
L2_WEIGHT = 1e-4
DROPOUT = 0.2

all_predictions = []
model_name = 'DL_HRD_SNVCNV_intermediate_crossatt_gistic'

for j in range(5):
    print(f"\n{'='*60}")
    print(f"Fold {j+1}/5")
    print(f"{'='*60}")

    df = pd.read_csv(cwd + f"HRDfolds{j+1}.csv")
    df = df.rename(columns={"Patient ID": 'bcr_patient_barcode'})
    df = df.sort_values(by='bcr_patient_barcode')
    df = df[df['bcr_patient_barcode'].isin(common_samples)]
    df = df.drop_duplicates()
    index_mapping = sample_df[['bcr_patient_barcode']].reset_index()
    small_df_filtered = df[df['bcr_patient_barcode'].isin(index_mapping['bcr_patient_barcode'])]
    small_df_ordered = index_mapping.merge(small_df_filtered, on='bcr_patient_barcode', how='inner')
    small_df_ordered = small_df_ordered.set_index('index').sort_index()
    df = small_df_ordered.drop_duplicates()
    mapping = {"HRD_positive": 1, "HRD_negative": 0}
    df['labels_int'] = df['HRD_binary_paper'].map(mapping)
    train_df = df[df["split"] == "train"]
    train_msih = train_df[train_df["HRD_binary_paper"] == "HRD_positive"]
    train_nonmsih = train_df[train_df["HRD_binary_paper"] == "HRD_negative"]
    train_resampled_nonmsih = train_nonmsih.sample(n=len(train_msih), random_state=SEED)
    train_df = pd.concat([train_msih, train_resampled_nonmsih], axis=0).sample(frac=1, random_state=SEED)
    train = train_df.index.tolist()
    test_df = df[df["split"] == "test"].sample(frac=1, random_state=SEED)
    test = test_df.index.tolist()
    val_df = df[df["split"] == "validation"]
    val_msih = val_df[val_df["HRD_binary_paper"] == "HRD_positive"]
    val_nonmsih = val_df[val_df["HRD_binary_paper"] == "HRD_negative"]
    val_resampled_nonmsih = val_nonmsih.sample(n=len(val_msih), random_state=SEED)
    val_df = pd.concat([val_msih, val_resampled_nonmsih], axis=0).sample(frac=1, random_state=SEED)
    val = val_df.index.tolist()

    X_cnv_arr = np.array(all_cancers_df).astype(float)
    X_cnv_arr_cptac = np.array(all_cancers_df_cptac).astype(float)
    val_cptac = list(range(len(sample_df_cptac)))

    y_label_fold = np.array(df['labels_int'].tolist())
    y_label_loader = DatasetsUtils.Map.FromNumpy(y_label_fold, tf.float32)
    cnv_loader = DatasetsUtils.Map.FromNumpy(X_cnv_arr, tf.float32)

    ds_train = tf.data.Dataset.from_tensor_slices(train)
    ds_train = ds_train.apply(DatasetsUtils.Apply.SubSample(batch_size=128, ds_size=len(train)))
    def process_train_batch(x):
        sample_idx, dropout_mask = index_loader(x)
        return (
            (
                five_p_loader(sample_idx, dropout_mask),
                three_p_loader(sample_idx, dropout_mask),
                ref_loader(sample_idx, dropout_mask),
                alt_loader(sample_idx, dropout_mask),
                strand_loader(sample_idx, dropout_mask),
                cnv_loader(sample_idx)
            ),
            y_label_loader(sample_idx)
        )
    ds_train = ds_train.map(process_train_batch)
    ds_train = ds_train.prefetch(1)

    ds_valid = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(val), three_p_loader_eval(val), ref_loader_eval(val), alt_loader_eval(val), strand_loader_eval(val), tf.gather(X_cnv_arr, val)), tf.gather(y_label_fold, val)))
    ds_valid = ds_valid.batch(len(val), drop_remainder=False)

    ds_test = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(test), three_p_loader_eval(test), ref_loader_eval(test), alt_loader_eval(test), strand_loader_eval(test), tf.gather(X_cnv_arr, test)), tf.gather(y_label_fold, test)))
    ds_test = ds_test.batch(len(test), drop_remainder=False)

    ds_test_cptac = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval_cptac(val_cptac), three_p_loader_eval_cptac(val_cptac), ref_loader_eval_cptac(val_cptac), alt_loader_eval_cptac(val_cptac), strand_loader_eval_cptac(val_cptac), tf.gather(X_cnv_arr_cptac, val_cptac)), tf.gather(y_label_cptac, val_cptac)))
    ds_test_cptac = ds_test_cptac.batch(len(val_cptac), drop_remainder=False)

    cnv_input = tf.keras.Input(shape=(N_CN_FEATURES,), name="cnv_input")
    x = tf.keras.layers.GaussianNoise(0.1)(cnv_input)
    encoded = tf.keras.layers.Dense(LATENT_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT), name="encoded_dense")(x)
    encoded = tf.keras.layers.BatchNormalization(name="encoded_bn")(encoded)
    encoded = tf.keras.layers.Dropout(DROPOUT, name="encoded")(encoded)
    cnv_encoder = Model(cnv_input, encoded, name="cnv_encoder")
    cnv_encoder.load_weights(os.path.join(WEIGHTS_DIR, f"HRD_ae_encoder_weights_fold{j+1}.h5"))
    cnv_encoder.trainable = False

    sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=128)
    mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=1, mil_hidden=(256, 128), attention_layers=[], dropout=dropout, instance_dropout=dropout, regularization=.1, input_dropout=dropout)

    with open(os.path.join(WEIGHTS_DIR, f"HRD_attMIL_weights_fold{j+1}.pkl"), "rb") as f:
        mil.model.set_weights(pickle.load(f))

    first_dense_idx = next(i for i, layer in enumerate(mil.model.layers) if isinstance(layer, tf.keras.layers.Dense))
    emb_layer = mil.model.layers[first_dense_idx - 1]
    mil_encoder = Model(mil.model.input, emb_layer.output, name="mil_encoder")
    mil_encoder.trainable = False

    mil_inputs = mil.model.inputs
    mil_emb = mil_encoder(mil_inputs)
    cnv_emb = cnv_encoder(cnv_input)

    # Intermediate fusion via bidirectional cross-attention
    mil_seq = tf.expand_dims(mil_emb, 1)
    cnv_seq = tf.expand_dims(cnv_emb, 1)
    mh_mil_to_cnv = MultiHeadAttention(num_heads=4, key_dim=64, name="mil_to_cnv_attn")
    mh_cnv_to_mil = MultiHeadAttention(num_heads=4, key_dim=64, name="cnv_to_mil_attn")
    ctx_mil = mh_mil_to_cnv(query=mil_seq, key=cnv_seq, value=cnv_seq)
    ctx_cnv = mh_cnv_to_mil(query=cnv_seq, key=mil_seq, value=mil_seq)
    ctx_mil = tf.squeeze(ctx_mil, axis=1)
    ctx_cnv = tf.squeeze(ctx_cnv, axis=1)

    x = Concatenate()([mil_emb, cnv_emb, ctx_mil, ctx_cnv])
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation="sigmoid", name="hrd_pred")(x)

    fusion_model = tf.keras.Model(inputs=mil_inputs + [cnv_input], outputs=output, name="fusion_model")
    fusion_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.002, patience=5, mode='max', restore_best_weights=True, start_from_epoch=1)]

    fusion_model.fit(ds_train, steps_per_epoch=math.ceil(len(train) / BATCH_SIZE), validation_data=ds_valid, epochs=20, callbacks=callbacks)

    cnv_encoder.trainable = True
    mil_encoder.trainable = True

    fusion_model.compile(optimizer=tf.keras.optimizers.Adam(1e-7), loss="binary_crossentropy", metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_BE', min_delta=0.002, patience=5, mode='min', restore_best_weights=True, start_from_epoch=1)]
    fusion_model.fit(ds_train, steps_per_epoch=math.ceil(len(train) / BATCH_SIZE), validation_data=ds_valid, epochs=10, callbacks=callbacks)

    fusion_model.compile(optimizer=tf.keras.optimizers.Adam(1e-6), loss="binary_crossentropy", metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_BE', min_delta=0.002, patience=5, mode='min', restore_best_weights=True, start_from_epoch=1)]
    fusion_model.fit(ds_train, steps_per_epoch=math.ceil(len(train) / BATCH_SIZE), validation_data=ds_valid, epochs=10, callbacks=callbacks)

    fusion_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_BE', min_delta=0.002, patience=5, mode='min', restore_best_weights=True, start_from_epoch=1)]
    fusion_model.fit(ds_train, steps_per_epoch=math.ceil(len(train) / BATCH_SIZE), validation_data=ds_valid, epochs=10, callbacks=callbacks)

    y_pred = fusion_model.predict(ds_test)
    for idx, (score, label) in enumerate(zip(y_pred.flatten(), y_label_fold[test].flatten())):
        all_predictions.append({
            'fold': j,
            'patient_id': test_df['bcr_patient_barcode'].iloc[idx],
            'cohort': 'TCGA_test',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })
    print(f"  TCGA test: {len(y_pred)} predictions")

    y_pred_cptac = fusion_model.predict(ds_test_cptac)
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
output_dir = "predictions_DL_SNVCNV_intermediate_crossatt_gistic/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions_HRD_SNVCNV_intermediate_crossatt_gistic_all_folds.csv")
predictions_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Total predictions: {len(predictions_df)}")
print(f"Cohorts: {predictions_df['cohort'].unique().tolist()}")

for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    cohort_path = os.path.join(output_dir, f"predictions_HRD_SNVCNV_intermediate_crossatt_gistic_{cohort}.csv")
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
print(f"  --output-dir evaluation_results_DL_SNVCNV_intermediate_crossatt_gistic/")

# %%
