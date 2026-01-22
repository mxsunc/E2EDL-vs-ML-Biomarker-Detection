#%%
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from mutationMIL.Sample_MIL2 import InstanceModels, RaggedModels
from tensorflow.KerasLayers import Losses, Metrics
from tensorflow import DatasetsUtils
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, 
    MultiHeadAttention, Concatenate, Model
)
import math
from experiments.helpers.align_dfs import align_dfB_to_dfA
from sklearn.metrics import roc_curve

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

def asfloat32(x):
    return np.asarray(x, dtype=np.float32)

# load SNV, INDEL data TCGA 
D, samples, sample_df = pickle.load(open('.../controlled_filters_combined_HRDwithlungucec_data_finished_20_pos.pkl', 'rb'))

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
dropout = 0.0
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

y_label = samples['class'][:, 1][:, np.newaxis]
y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)

# load SNV, INDEL data CPTAC
D_cptac, samples_cptac, sample_df_cptac = pickle.load(open('.../HRDwithlungucec_cptac3_data_finished_20_pos.pkl', 'rb'))

sample_df_cptac = sample_df_cptac.drop(columns=["index"])
sample_df_cptac = sample_df_cptac.drop_duplicates(["Patient ID"])

strand_emb_mat_cptac = np.concatenate([np.zeros(2)[np.newaxis, :], np.diag(np.ones(2))], axis=0)
D_cptac['strand_emb'] = strand_emb_mat_cptac[D['strand']]

chr_emb_mat_cptac = np.concatenate([np.zeros(24)[np.newaxis, :], np.diag(np.ones(24))], axis=0)
D_cptac['chr_emb'] = chr_emb_mat_cptac[D['chr']]

indexes_cptac = [np.where(D_cptac['sample_idx'] == idx) for idx in range(sample_df_cptac.shape[0])]

five_p_cptac = np.array([D_cptac['seq_5p'][i] for i in indexes_cptac], dtype='object')
three_p_cptac = np.array([D_cptac['seq_3p'][i] for i in indexes_cptac], dtype='object')
ref_cptac = np.array([D_cptac['seq_ref'][i] for i in indexes_cptac], dtype='object')
alt_cptac = np.array([D_cptac['seq_alt'][i] for i in indexes_cptac], dtype='object')
strand_cptac = np.array([D_cptac['strand_emb'][i] for i in indexes_cptac], dtype='object')
dropout_cptac = .0
index_loader_cptac = DatasetsUtils.Map.FromNumpytoIndices([j for i in indexes_cptac for j in i], dropout=dropout_cptac)
five_p_loader_cptac = DatasetsUtils.Map.FromNumpyandIndices(five_p_cptac, tf.int16)
three_p_loader_cptac = DatasetsUtils.Map.FromNumpyandIndices(three_p_cptac, tf.int16)
ref_loader_cptac = DatasetsUtils.Map.FromNumpyandIndices(ref_cptac, tf.int16)
alt_loader_cptac = DatasetsUtils.Map.FromNumpyandIndices(alt_cptac, tf.int16)
strand_loader_cptac = DatasetsUtils.Map.FromNumpyandIndices(strand_cptac, tf.float32)

five_p_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(five_p_cptac, tf.int16)
three_p_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(three_p_cptac, tf.int16)
ref_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(ref_cptac, tf.int16)
alt_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(alt_cptac, tf.int16)
strand_loader_eval_cptac = DatasetsUtils.Map.FromNumpy(strand_cptac, tf.float32)

y_label_cptac = samples_cptac['class'][:, 1][:, np.newaxis]

# load CNV TCGA
gene_pos_df = pd.read_csv("...s/CNV_pos_embeddings_100k.csv")
chrom_order = {**{str(i): i for i in range(1,23)}, "X": 23, "Y": 24, "MT": 25}
genes_ordered = gene_pos_df["gene"].tolist()

input_path = ".../tcga_cnv_seg_matrix.tsv"
all_cancers_df = pd.read_csv(input_path, sep="\t")
all_cancers_df = all_cancers_df.dropna()

all_cancers_df = all_cancers_df.set_index("patient_id")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()
all_cancers_df["bcr_patient_barcode"] = all_cancers_df.index
samples_list_snv = sample_df['bcr_patient_barcode'].tolist()
samples_list_cnv = all_cancers_df["bcr_patient_barcode"].tolist()
common_samples = list(set(samples_list_snv)&set(samples_list_cnv))
all_cancers_df = all_cancers_df[all_cancers_df["bcr_patient_barcode"].isin(common_samples)]
all_cancers_df = all_cancers_df.sort_values(by="bcr_patient_barcode")
all_cancers_df = all_cancers_df.reset_index(drop=True)
all_cancers_df.index = all_cancers_df["bcr_patient_barcode"]
l = samples["bcr_patient_barcode"].tolist()
all_cancers_df = all_cancers_df.reindex(l)
all_cancers_df= all_cancers_df.reset_index(drop=True)

shared = [g for g in genes_ordered if g in all_cancers_df.columns]
all_cancers_df = all_cancers_df.loc[:, shared].copy()
chrom_order = {**{str(i): i for i in range(1,23)}, "X": 23, "Y": 24, "MT": 25}
genes_ordered = gene_pos_df["gene"].tolist()

input_path = ".../cptac_cnv_seg_matrix.tsv"
all_cancers_df_cptac = pd.read_csv(input_path, sep="\t")
all_cancers_df_cptac=all_cancers_df_cptac.dropna()
all_cancers_df_cptac = all_cancers_df_cptac.rename(columns={"PATIENT":"Patient ID"})
samples_list_snv_cptac = sample_df_cptac['Patient ID'].tolist()
samples_list_cnv_cptac = all_cancers_df_cptac['Patient ID'].tolist()
common_samples_cptac = list(set(samples_list_snv_cptac)&set(samples_list_cnv_cptac))

all_cancers_df_cptac = all_cancers_df_cptac[all_cancers_df_cptac['Patient ID'].isin(common_samples_cptac)]
all_cancers_df_cptac = all_cancers_df_cptac.sort_values(by='Patient ID')
all_cancers_df_cptac.index= all_cancers_df_cptac['Patient ID']
l = samples_cptac["Patient ID"].tolist()
all_cancers_df_cptac = all_cancers_df_cptac.reindex(l)
all_cancers_df= all_cancers_df.reset_index(drop=True)
all_cancers_df = all_cancers_df.iloc[:,:-1]
all_cancers_df_cptac = align_dfB_to_dfA(all_cancers_df,all_cancers_df_cptac)

df_hrd_cptac = pd.read_excel(".../3_CPTAC_scarHRD_us.xlsx")
df_hrd_cptac = df_hrd_cptac.dropna(subset=["HRD_Binary_us"]).copy()
df_hrd_cptac["HRD_Binary_us"] = df_hrd_cptac["HRD_Binary_us"].map({
    "WT": 0,
    "MUT": 1
})

sample_df_cptac["HRD_Binary"] = df_hrd_cptac["HRD_Binary_us"].map({
    "WT": 0,
    "MUT": 1
})

accu = []
f1s = []
precisions = []
recalls = []
thresholds_prs = []
tprs = []
fprs = []
thresholdss = []
auc_scores = []
auc_scores_pr = []

tps, fps, tns, fns = [], [], [], []
sensitivities, specificities = [], []

accu_cptac = []
f1s_cptac = []
precisions_cptac = []
recalls_cptac = []
thresholds_prs_cptac = []
tprs_cptac = []
fprs_cptac = []
thresholdss_cptac = []
auc_scores_cptac = []
auc_scores_pr_cptac = []
tps_cptac, fps_cptac, tns_cptac, fns_cptac = [], [], [], []
sensitivities_cptac, specificities_cptac = [], []

weights = []
test_idx = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_BE', min_delta=0.002, patience=5, mode='min', restore_best_weights=True,start_from_epoch = 1)]
losses = [Losses.BinaryCrossEntropy(from_logits=True)]

for j in range(5):
    df = pd.read_csv(".../HRDfolds"+str(j+1)+"_withlungucec.csv")
    df = df.rename(columns={"Patient ID":'bcr_patient_barcode'})
    df = df.sort_values(by='bcr_patient_barcode')
    df = df[df['bcr_patient_barcode'].isin(common_samples)]
    df = df.drop_duplicates()
    index_mapping = sample_df[['bcr_patient_barcode']].reset_index()
    small_df_filtered = df[df['bcr_patient_barcode'].isin(index_mapping['bcr_patient_barcode'])]
    small_df_ordered = index_mapping.merge(small_df_filtered, on='bcr_patient_barcode', how='inner')
    small_df_ordered = small_df_ordered.set_index('index').sort_index()
    df = small_df_ordered.drop_duplicates()
    mapping = {"HRD_positive":1,"HRD_negative":0}
    df['labels_int'] = df['HRD_status'].map(mapping)
    train_df = df[df["split"]=="train"]
    train_msih = train_df[train_df["HRD_binary_paper"]=="HRD_positive"]
    train_nonmsih  = train_df[train_df["HRD_binary_paper"]=="HRD_negative"]
    train_resampled_nonmsih = train_nonmsih.sample(n=len(train_msih))
    train_df = pd.concat([train_msih, train_resampled_nonmsih], axis=0)
    train_df = train_df.sample(frac=1)
    train = train_df.index.tolist()
    labels_train = np.array(train_df['labels_int'].tolist())
    test_df = df[df["split"]=="test"] 
    test_df = test_df.sample(frac=1)
    test = test_df.index.tolist()
    labels_test = np.array(test_df['labels_int'].tolist())
    val_df = df[df["split"]=="validation"]
    val_msih  = val_df[val_df["HRD_binary_paper"]=="HRD_positive"]
    val_nonmsih  = val_df[val_df["HRD_binary_paper"]=="HRD_negative"]
    val_resampled_nonmsih = val_nonmsih.sample(n=len(val_msih))
    val_df = pd.concat([val_msih, val_resampled_nonmsih], axis=0)
    val_df = val_df.sample(frac=1)
    labels_val = np.array(val_df['labels_int'].tolist())
    val = val_df.index.tolist()
    X_cnv_arr = np.array(all_cancers_df)
    X_cnv_arr = X_cnv_arr.astype(float)
    X_cnv_arr_cptac = np.array(all_cancers_df_cptac)
    X_cnv_arr_cptac = X_cnv_arr_cptac.astype(float)
    val_cptac = sample_df_cptac.index.tolist()
    labels_all = np.array(df['labels_int'].tolist())
    labels_all_cptac = np.array(sample_df_cptac['HRD_Binary'].tolist())

    BATCH_SIZE = 64
    HALF = BATCH_SIZE // 2
    AUTOTUNE = tf.data.AUTOTUNE
    N_CN_FEATURES = len(all_cancers_df_cptac.columns)
    LATENT_DIM  = 512
    L2_WEIGHT = 0.05
    DROPOUT = 0.05

    cnv_loader = DatasetsUtils.Map.FromNumpy(X_cnv_arr, tf.float32)

    ds_train = tf.data.Dataset.from_tensor_slices(train)
    ds_train = ds_train.apply(DatasetsUtils.Apply.SubSample(batch_size=128, ds_size=len(train)))

    ds_train = ds_train.map(lambda x: ((
            index_loader(x),)),)

    ds_train = ds_train.map(lambda x: ((five_p_loader(x[0], x[1]),
                                                three_p_loader(x[0], x[1]),
                                                ref_loader(x[0], x[1]),
                                                alt_loader(x[0], x[1]),
                                                strand_loader(x[0], x[1]),
                                                cnv_loader(x[0])
                                            ),
                                           y_label_loader(x[0]),
                                           ))
    ds_train.prefetch(1)

    ds_valid = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(val),
                                               three_p_loader_eval(val),
                                               ref_loader_eval(val),
                                               alt_loader_eval(val),
                                               strand_loader_eval(val),
                                               tf.gather(X_cnv_arr, val)
                                            ),
                                           tf.gather(y_label, val),
                                           ))
    ds_valid = ds_valid.batch(len(val), drop_remainder=False)

    ds_test = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(test),
                                               three_p_loader_eval(test),
                                               ref_loader_eval(test),
                                               alt_loader_eval(test),
                                               strand_loader_eval(test),
                                               tf.gather(X_cnv_arr, test)
                                            ),
                                           tf.gather(y_label, test),
                                           ))
    ds_test = ds_test.batch(len(test), drop_remainder=False)

    ds_test_cptac = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval_cptac(val_cptac),
                                               three_p_loader_eval_cptac(val_cptac),
                                               ref_loader_eval_cptac(val_cptac),
                                               alt_loader_eval_cptac(val_cptac),
                                               strand_loader_eval_cptac(val_cptac),
                                               tf.gather(X_cnv_arr_cptac, val_cptac)
                                            ),
                                           tf.gather(y_label_cptac, val_cptac),
                                           ))
    ds_test_cptac = ds_test_cptac.batch(len(val_cptac), drop_remainder=False)


    STEPS_PER_EPOCH = math.ceil(len(train) / BATCH_SIZE)
    cnv_input = tf.keras.Input(shape=(N_CN_FEATURES,), name="cnv_input")

    x        = tf.keras.layers.GaussianNoise(0.1)(cnv_input)
    encoded  = tf.keras.layers.Dense(LATENT_DIM, activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
                    name="encoded_dense")(x)
    encoded  = tf.keras.layers.BatchNormalization(name="encoded_bn")(encoded)
    encoded  = tf.keras.layers.Dropout(DROPOUT, name="encoded")(encoded)
    cnv_encoder = Model(cnv_input, encoded, name="cnv_encoder")

    cnv_encoder.load_weights(".../weights/fold"+str(j+1)+"_ae_encoder_weights_seq.h5")
    cnv_encoder.trainable = False

    sequence_encoder = InstanceModels.VariantSequence(20, 4, 2,[8, 8, 8, 8], fusion_dimension=128)
    mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=1, mil_hidden=(256, 128), attention_layers=[], dropout=dropout, instance_dropout=dropout, regularization=.1, input_dropout=dropout)

    first_dense_idx = next(
        i for i, layer in enumerate(mil.model.layers)
        if isinstance(layer, tf.keras.layers.Dense)
    )
    emb_layer = mil.model.layers[first_dense_idx - 1]
    mil_encoder = Model(mil.model.input, emb_layer.output, name="mil_encoder")

    with open(".../weights/HRDwithlungucec_mil_encoder_weights_fold"+str(j+1)+".pkl", "rb") as f:
        weights = pickle.load(f) 
    mil_encoder.set_weights(weights)
    mil_encoder.trainable = False  

    seq_in = mil.model.input                   
    cnv_in = cnv_encoder.input

    mil_emb = mil_encoder(seq_in)                
    cnv_emb = cnv_encoder(cnv_in)                

    mil_inputs = mil.model.inputs

    mil_emb = mil_encoder(mil_inputs)
    cnv_emb = cnv_encoder(cnv_input)

    x = tf.keras.layers.Concatenate()([mil_emb, cnv_emb])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="hrd_pred")(x)

    fusion_model = tf.keras.Model(
        inputs=mil_inputs + [cnv_input],
        outputs=output,
        name="fusion_model"
    )
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy']
    )

    fusion_model.fit(
        ds_train,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=ds_valid,
        epochs=20,
        callbacks=callbacks
    )

    y_pred = fusion_model.predict(ds_test)
    fpr, tpr, thresholds = roc_curve(y_label[test], y_pred)
    youden_j = tpr - fpr
    best_threshold_index = np.argmax(youden_j)
    best_threshold = thresholds[best_threshold_index]
    y_pred_binary = (y_pred >= best_threshold).astype(int)

    y_pred_cptac = fusion_model.predict(ds_test_cptac)
    y_pred_binary_cptac = (y_pred_cptac >= best_threshold).astype(int)