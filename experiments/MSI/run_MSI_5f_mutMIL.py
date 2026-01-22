import numpy as np
import tensorflow as tf
from mutationMIL.Sample_MIL import InstanceModels, RaggedModels
from mutationMIL.KerasLayers import Losses, Metrics
from mutationMIL import DatasetsUtils
import pandas as pd
import pickle
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

cwd = "..." 

D, samples, sample_df = pickle.load(open(cwd + '/controlled_allmut_multi_msi_consensus_data_finished_20.pkl', 'rb'))

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
dropout = .4
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

weights = []
test_idx = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_BE', min_delta=0.002, patience=50, mode='min', restore_best_weights=True)]
losses = [Losses.BinaryCrossEntropy(from_logits=True)]
samples_list = sample_df['bcr_patient_barcode'].tolist()

for j in range(5):
    df = pd.read_csv(cwd+"/MSIfolds"+str(j+1)+".csv")
    df = df.sort_values(by='bcr_patient_barcode')
    df = df[df['bcr_patient_barcode'].isin(samples_list)]
    df.reset_index(inplace=True, drop=True)
    train_df = df[df["split"]=="train"]
    train_df = train_df.sample(frac=1)
    train = train_df.index.tolist()
    test_df = df[df["split"]=="test"] 
    test_df = test_df.sample(frac=1)
    test = test_df.index.tolist()
    val_df = df[df["split"]=="validation"]
    val_df = val_df.sample(frac=1)
    val = val_df.index.tolist()
    y_label_for_roc = y_label[:, 0][:, np.newaxis]

    ds_train = tf.data.Dataset.from_tensor_slices(train)
    ds_train = ds_train.apply(DatasetsUtils.Apply.SubSample(batch_size=128, ds_size=len(train)))
    ds_train = ds_train.map(lambda x: ((
            index_loader(x),)),)
    ds_train = ds_train.map(lambda x: ((five_p_loader(x[0], x[1]),
                                                three_p_loader(x[0], x[1]),
                                                ref_loader(x[0], x[1]),
                                                alt_loader(x[0], x[1]),
                                                strand_loader(x[0], x[1]),
                                            ),
                                           y_label_loader(x[0]),
                                           ))
    ds_train.prefetch(1)
    ds_valid = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(val),
                                               three_p_loader_eval(val),
                                               ref_loader_eval(val),
                                               alt_loader_eval(val),
                                               strand_loader_eval(val),
                                            ),
                                           tf.gather(y_label, val),
                                           ))
    ds_valid = ds_valid.batch(len(val), drop_remainder=False)
    sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=128)
    mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=y_label.shape[-1], mil_hidden=(256, 128), attention_layers=[], dropout=.5, instance_dropout=.5, regularization=.05, input_dropout=dropout)
    mil.model.compile(loss=losses,
                          metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    mil.model.fit(ds_train,
                      steps_per_epoch=10,
                      validation_data=ds_valid,
                      epochs=200,
                      callbacks=callbacks,)
    eval = mil.model.evaluate(ds_valid)
    weights = mil.model.get_weights()
    with open(cwd+'/MSI_allmut_20_model_weights_fold'+str(j+1)+'.pkl', 'wb') as f:
        pickle.dump(weights, f)