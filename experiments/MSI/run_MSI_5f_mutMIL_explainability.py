#%%
import numpy as np
import tensorflow as tf
from mutationMIL.Sample_MIL import InstanceModels, RaggedModels
from mutationMIL.KerasLayers import Losses, Metrics
from mutationMIL import DatasetsUtils
import pandas as pd
import pickle
from matplotlib.colors import LinearSegmentedColormap, mcolors
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from helpers.mutation_classification import generate_mutation_classes, decode_sequence, classify_mutations, decode_sequence, reverse_complement, classify_indels_detailed, plot_sequence_logos_for_cluster

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

cwd = "..." 
dropout = 0.4
D, samples, sample_df = pickle.load(open(cwd + '/controlled_filters_multi_msi_consensus_data_finished_20_pos.pkl', 'rb'))

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

five_p_loader_eval = DatasetsUtils.Map.FromNumpy(five_p, tf.int16)
three_p_loader_eval = DatasetsUtils.Map.FromNumpy(three_p, tf.int16)
ref_loader_eval = DatasetsUtils.Map.FromNumpy(ref, tf.int16)
alt_loader_eval = DatasetsUtils.Map.FromNumpy(alt, tf.int16)
strand_loader_eval = DatasetsUtils.Map.FromNumpy(strand, tf.float32)

y_label = samples['class'][:, 0][:, np.newaxis]
y_label_loader = DatasetsUtils.Map.FromNumpy(y_label, tf.float32)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_BE', min_delta=0.002, patience=50, mode='min', restore_best_weights=True)]
losses = [Losses.BinaryCrossEntropy(from_logits=True)]
samples_list = sample_df['bcr_patient_barcode'].tolist()

sequence_encoder = InstanceModels.VariantSequence(20, 4, 2, [8, 8, 8, 8], fusion_dimension=128)
mil = RaggedModels.MIL(instance_encoders=[sequence_encoder.model], sample_encoders=[], heads=y_label.shape[-1], mil_hidden=(256, 128), attention_layers=[], dropout=.5, instance_dropout=.5, regularization=.05, input_dropout=dropout)
mil.model.compile(loss=losses,
                          metrics=[Metrics.BinaryCrossEntropy(from_logits=True), 'accuracy'],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

loaded_weights = []
with open(cwd+'/MSI_filters_20_model_weights_fold4.pkl', 'rb') as f:
        loaded_weights.append(pickle.load(f))
mil.model.set_weights(loaded_weights[0])

cancer_logos = []
cancer_positions =[]
cancer_features_norm = []
cancer_features_notnorm = []
mutation_attention = []
# set the layers that encodes single mutations to get mutation level features
model_feature = Model(inputs=mil.model.inputs, outputs=mil.model.layers[-11].output)
ds_all = tf.data.Dataset.from_tensor_slices((np.unique(D["sample_idx"]), y_label))
ds_all = ds_all.batch(len(y_label), drop_remainder=False)
ds_all = tf.data.Dataset.from_tensor_slices(((five_p_loader_eval(np.unique(D["sample_idx"])),
                                               three_p_loader_eval(np.unique(D["sample_idx"])),
                                               ref_loader_eval(np.unique(D["sample_idx"])),
                                               alt_loader_eval(np.unique(D["sample_idx"])),
                                               strand_loader_eval(np.unique(D["sample_idx"])),
                                            ),
                                           tf.gather(y_label, np.unique(D["sample_idx"])),
                                           ))
ds_all = ds_all.batch(len(np.unique(D["sample_idx"])), drop_remainder=False)
# get features
features = model_feature.predict(ds_all).flat_values.numpy()

# get corresponfing attention values
attention_scores = mil.attention_model.predict(ds_all).flat_values.numpy()
attention_scores_flat = np.array([item for sublist in attention_scores for item in sublist])

# select 5000 random mutations 
if len(features) > 5000:
        selected_indices = np.random.choice(len(features), 5000, replace=False)
        features = features[selected_indices]
        attention_scores_flat = attention_scores_flat[selected_indices]

mutation_attention.append(attention_scores_flat)
cancer_features_notnorm.append(features)

min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
features_normalized = (features - min_vals) / (max_vals - min_vals + 1e-5)
cancer_features_norm.append(features_normalized)

# get sequences of each block 
top_five_sequences = np.array(D['seq_5p'])[selected_indices]
top_three_sequences = np.array(D['seq_3p'])[selected_indices]
top_reference_sequences = np.array(D['seq_ref'])[selected_indices]
top_alteration_sequences = np.array(D['seq_alt'])[selected_indices]
top_positions = np.array(D['pos_float'])[selected_indices]

# deconde numbers into letters
decoded_five_sequences = decode_sequence(top_five_sequences)
decoded_three_sequences = decode_sequence(top_three_sequences)
decoded_reference = decode_sequence(top_reference_sequences)
decoded_alteration = decode_sequence(top_alteration_sequences)
cancer_positions.append(top_positions)
cancer_logos.append([decoded_five_sequences,decoded_three_sequences,decoded_reference,decoded_alteration])

five_sequences = [x[0] for x in cancer_logos]
five_sequences = [item for sublist in five_sequences for item in sublist]
three_sequences = [x[1] for x in cancer_logos]
three_sequences = [item for sublist in three_sequences for item in sublist]
ref_sequences = [x[2] for x in cancer_logos]
ref_sequences = [item for sublist in ref_sequences for item in sublist]
alt_sequences = [x[3] for x in cancer_logos]
alt_sequences = [item for sublist in alt_sequences for item in sublist]
posinces = [item for sublist in cancer_positions for item in sublist]

all_mutation_features = [item for sublist in cancer_features_notnorm for item in sublist]
all_mutation_features = [subarray for subarray in all_mutation_features]
all_mutation_attention = np.array([item for sublist in mutation_attention for item in sublist])
all_positions = np.array([item for sublist in cancer_positions for item in sublist])
names_cancer_types = sample_df['PCR'].unique().tolist()
names_cancer_types_per_mutation = []
for i, sublist in enumerate(cancer_features_notnorm):
    for item in sublist:
        names_cancer_types_per_mutation.append(names_cancer_types[i])

indel_indices = [
    i for i, (ref, alt) in enumerate(zip(ref_sequences, alt_sequences))
    if all(char == '-' for char in ref) or all(char == '-' for char in alt)
]

# filter mutation features and attention values for indels
indel_mutation_features = [all_mutation_features[i] for i in indel_indices]
indel_mutation_attention = all_mutation_attention[indel_indices]

# filter sequences for visualization or further analysis
indel_five_sequences = [five_sequences[i] for i in indel_indices]
indel_three_sequences = [three_sequences[i] for i in indel_indices]
indel_ref_sequences = [ref_sequences[i] for i in indel_indices]
indel_alt_sequences = [alt_sequences[i] for i in indel_indices]
indel_positions = [posinces[i] for i in indel_indices]

# heatmap of mutation features
colors = ["white","#83c7e9","#185d85", "#bc1e2d", "#f4969d"]
cmap_name = 'custom_coolwarm'
custom_coolwarm = LinearSegmentedColormap.from_list(cmap_name, colors)
features_array = np.array(all_mutation_features)
feature_array_indel = np.array(indel_mutation_features)
plt.rcParams['figure.dpi'] = 600
plt.imshow(features_array, aspect='auto', cmap=custom_coolwarm)
plt.title('Heatmap of Features')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.gcf().patch.set_alpha(0)
plt.gca().patch.set_alpha(0)
plt.tight_layout()
plt.show()

#cluster features
kmeans = KMeans(n_clusters=7, random_state=42)
clusters = kmeans.fit_predict(features_array)
five_prime_ends_array = np.array(five_sequences)
three_prime_ends_array = np.array(three_sequences)
ref_seqs_array = np.array(ref_sequences)
alt_seqs_array = np.array(alt_sequences)

mean_attentions = np.array([np.mean(all_mutation_attention[clusters == i]) for i in range(7)])

clusters_order = np.argsort(mean_attentions)
sorted_mean_attention = mean_attentions[clusters_order]

sorted_clusters = np.zeros_like(clusters)
for new_order, cluster_id in enumerate(clusters_order):
    sorted_clusters[clusters == cluster_id] = new_order

# sort features by cluster
sort_idx = np.argsort(sorted_clusters)
sorted_features_array = features_array[sort_idx]
sorted_five_prime_ends = five_prime_ends_array[sort_idx]
sorted_three_prime_ends = three_prime_ends_array[sort_idx]
sorted_ref_seqs = ref_seqs_array[sort_idx]
sorted_alt_seqs = alt_seqs_array[sort_idx]
sorted_mutation_attention = all_mutation_attention[sort_idx]

cluster_boundaries = np.where(np.diff(sorted_clusters[sort_idx]))[0]

length_of_sorted_features_array = len(sorted_features_array)
cluster_assignment = []
for i in range(length_of_sorted_features_array):
    for cluster_number, boundary in enumerate(cluster_boundaries, start=1):
        if i < boundary:
            cluster_assignment.append(cluster_number)
            break
    else:
        cluster_assignment.append(cluster_number + 1)

# heatmap of clustered mutation features
plt.figure(figsize=(7.5, 6))
plt.rcParams['figure.dpi'] = 600
plt.imshow(sorted_features_array, aspect='auto', alpha = 0.9,cmap = custom_coolwarm)  
plt.colorbar()  
plt.title('Heatmap of Features Array')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.gcf().patch.set_alpha(0)
plt.gca().patch.set_alpha(0)
for boundary in cluster_boundaries:
        plt.axhline(y=boundary, color='#252525', linestyle=(0, (5, 4)),linewidth=1.0)
plt.title('Heatmap of Features')
plt.xlabel('Feature Dimension')
plt.ylabel('Mutations')
plt.show()
plt.close()

unique_clusters = np.unique(sorted_clusters)
adjusted_boundaries = cluster_boundaries + 1
final_boundaries = np.append(adjusted_boundaries, len(sorted_clusters[sort_idx]))

# splitting each array using the calculated boundaries
features_by_cluster = np.split(sorted_features_array, final_boundaries[:-1])
five_prime_ends_by_cluster = np.split(sorted_five_prime_ends, final_boundaries[:-1])
three_prime_ends_by_cluster = np.split(sorted_three_prime_ends, final_boundaries[:-1])
ref_seqs_by_cluster = np.split(sorted_ref_seqs, final_boundaries[:-1])
alt_seqs_by_cluster = np.split(sorted_alt_seqs, final_boundaries[:-1])
mutation_attention_by_cluster = np.split(sorted_mutation_attention, final_boundaries[:-1])

cluster_dataframes = []
for i in range(len(features_by_cluster)):
    # Prepare a dictionary to hold data for DataFrame creation
    data = {
        "5_prime": five_prime_ends_by_cluster[i],
        "3_prime": three_prime_ends_by_cluster[i],
        "ref": ref_seqs_by_cluster[i],
        "alt": alt_seqs_by_cluster[i],
        "all_mutation_attention": mutation_attention_by_cluster[i],
    }

    for j in range(128): # feature dimension
        data[f'feature_{j}'] = features_by_cluster[i][:, j]

    df = pd.DataFrame(data)
    cluster_dataframes.append(df)

mutation_classes = generate_mutation_classes()
reverse_complements = [reverse_complement(mut) for mut in mutation_classes]

from builtins import min
for cluster_index in range(len(cluster_dataframes)):
        
        five_prime_seqs = cluster_dataframes[cluster_index]["5_prime"]
        ref_seqs = cluster_dataframes[cluster_index]["ref"]
        alt_seqs = cluster_dataframes[cluster_index]["alt"]
        three_prime_seqs = cluster_dataframes[cluster_index]["3_prime"]

        # plot seuqence logos for the cluster separated by sequence block
        plot_sequence_logos_for_cluster('cancer_name',0,five_prime_seqs, ref_seqs, alt_seqs, three_prime_seqs)

        # classify mutations
        sbs_mutations = []
        for x in range(len(five_prime_seqs)):
            five_prime = five_prime_seqs[x][-1] 
            ref_base = ref_seqs[x][0]
            alt_base = alt_seqs[x][0] 
            three_prime = three_prime_seqs[x][0] 
            mutation_class = classify_mutations(five_prime, ref_base, alt_base, three_prime)
            sbs_mutations.append(mutation_class)

        cluster_dataframes[cluster_index]["mutation_class"] = ["SBS" if x != 'Other' else "InDel" for x in sbs_mutations]

        # separate mutations by SBSs and indels
        df_sbs = cluster_dataframes[cluster_index][cluster_dataframes[cluster_index]['mutation_class'] == 'SBS'].iloc[:, :]
        df_indel = cluster_dataframes[cluster_index][cluster_dataframes[cluster_index]['mutation_class'] == 'InDel'].iloc[:, :]

        mutation_dict = {}
        mutation_dict_small = {"C>A":0, "C>G":0,"C>T":0,"T>A":0,"T>C":0,"T>G":0}
    
        for i in range(len(mutation_classes)):
            mutation_class = mutation_classes[i]
            reverse_mutation_class = reverse_complements[i]
            count = 0
            for mutation in sbs_mutations:
                if mutation.startswith(mutation_class) or mutation.startswith(reverse_mutation_class):
                    count += 1
                    if "C>A" in mutation or "G>T" in mutation:
                        mutation_dict_small["C>A"] += 1
                    if "C>G" in mutation or "G>C" in mutation:
                        mutation_dict_small["C>G"] += 1
                    if "C>T" in mutation or "G>A" in mutation:
                        mutation_dict_small["C>T"] += 1
                    if "T>A" in mutation or "A>T" in mutation:
                        mutation_dict_small["T>A"] += 1
                    if "T>C" in mutation or "A>G" in mutation:
                        mutation_dict_small["T>C"] += 1
                    if "T>G" in mutation or "A>C" in mutation:
                        mutation_dict_small["T>G"] += 1
            count = count / len(sbs_mutations)
            mutation_dict[mutation_class] = count
            
        # plot SBS features for each cluster
        plt.rcParams['figure.dpi'] = 600
        colors = ['#42c2f5'] * 16 + ['black'] * 16 + ['#b51626'] * 16 + ['lightgrey'] * 16 + ['#a4d466'] * 16 + ['#e8b0b4'] * 16
        plt.figure(figsize=(13, 2))
        rgb_colors = [mcolors.to_rgb(color) for color in colors]
        bars = plt.bar(
            mutation_dict.keys(),
            mutation_dict.values(),
            color=colors)

        plt.ylabel('Probability')
        for idx, key in enumerate([x[0]+x[1]+x[4] for x in mutation_dict.keys()]):
            x_position = idx  
            y_position = -0.007  
            first_letter = key[0]
            middle_letter = key[1]
            last_letter = key[2]
            color = colors[idx]  

            plt.text(
                x_position, y_position,
                f"{first_letter}{middle_letter}{last_letter}", 
                color='black',
                ha='center', va='top',
                fontsize=8,
                rotation=90
            )
            # overlay the middle letter with its specific color
            plt.text(
                x_position, y_position-0.0007,
                " "+middle_letter+" ",
                color=color,
                ha='center', va='top', fontsize=10,fontweight='bold',
                rotation=90
            )

        plt.xticks([])
        plt.tight_layout()
        plt.xlim(-1, 96)
        plt.gca().patch.set_alpha(0)
        plt.show()
        plt.close()

        # plot indel features for each cluster
        colors_indels = ["#fcbe6e","#fd7f03","#afda87","#3a9e33","#fcc9b5","#fb8969","#f04435","#b51d18","#d0dfef","#93c2e1","#4c95c7","#1961aa","#e3e2ea", "#b2b5d9", "#8784b9", "#614199"] # 16 colors
        indel_mutation_dict = classify_indels_detailed(df_indel["5_prime"],df_indel["ref"],df_indel["alt"],df_indel["3_prime"],(len(df_sbs)+len(df_indel)))

        colors_indels = ["#fcbe6e"] * 6+["#fd7f03"] * 6+["#afda87"] * 6+["#3a9e33"]*6+["#fcc9b5"]*6+["#fb8969"]*6+["#f04435"]*6+["#b51d18"]*6+["#d0dfef"]*6+["#93c2e1"]*6+["#4c95c7"]*6+["#1961aa"]*6+["#e3e2ea"]+["#b2b5d9"]*2+["#8784b9"]*3+ ["#614199"]*6

        plt.figure(figsize=(10,2))
        bars = plt.bar(indel_mutation_dict.keys(), indel_mutation_dict.values(), color=colors_indels)
        plt.ylabel('Probability')
        plt.xticks(rotation=0)
        plt.tick_params(axis='x', pad=17)
        smal = ["1","2","3","4","5","6"]
        lon = ["0","1","2","3","4","5"]
        plt.gca().set_xticklabels(2*lon + 2*smal + 4*lon+4*smal+["1","1","2","1","2","3"]+smal)
        plt.tight_layout()
        plt.xlim(-1,83)
    
#%% 
# mutation level UMAP and T-SNE
model_feature = Model(inputs=mil.model.inputs, outputs=mil.model.layers[-11].output)
features = model_feature.predict(ds_all).flat_values.numpy()
mutation_patient_indices = D["sample_idx"].tolist()
mutation_labels = []
classes = samples["class"][:, 0][:, np.newaxis]
mutation_labels = [classes[x] for x in mutation_patient_indices]
features_label_0 = np.array([feat for feat, label in zip(features, mutation_labels) if label == [0.]])
features_label_1 = np.array([feat for feat, label in zip(features, mutation_labels) if label == [1.]])

selected_indices1 = np.random.choice(len(features_label_1), 1000, replace=False)
selected_indices0 = np.random.choice(len(features_label_0), 1000, replace=False)
features_label_0 = features_label_0[selected_indices0]
features_label_1 = features_label_1[selected_indices1]

tsne = TSNE(n_components=2, random_state=42)
feature_indices0 = range(1000)
feature_indices1 = range(1000,2000)
patient_features_2d = tsne.fit_transform(np.concatenate([features_label_0, features_label_1]))

plt.figure(figsize=(2.6,2.5))
unique_labels = np.unique([0.0,1.0])
plt.scatter(patient_features_2d[feature_indices0,0], patient_features_2d[feature_indices0,1],s=20, alpha=0.1, color='#f4969d')
plt.scatter(patient_features_2d[feature_indices1,0], patient_features_2d[feature_indices1,1],s=20, alpha=0.1, color='#83c7e9')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()
plt.close()

reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(np.concatenate([features_label_0, features_label_1]))

cancer_colors = {
 'MSIH': '#f4969d',
 'nonMSIH': '#83c7e9',
}

plt.figure(figsize=(2.6,2.5))
unique_labels = np.unique([0.0,1.0])
plt.scatter(embedding[feature_indices0,0], embedding[feature_indices0,1],s=20, alpha=0.1, color='#f4969d')
plt.scatter(embedding[feature_indices1,0], embedding[feature_indices1,1],s=20, alpha=0.1, color='#83c7e9')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()
plt.close()

#%%  
#patient level UMAP and T-SNE
model_feature = Model(inputs=mil.model.inputs, outputs=mil.model.layers[-5].output)
features = model_feature.predict(ds_all)

integer_labels = np.argmax(y_label, axis=1)
string_labels = sample_df["PCR"].tolist()

tsne = TSNE(n_components=2, random_state=42)
patient_features_2d = tsne.fit_transform(features)

cancer_colors = {
 'MSIH': '#f4969d',
 'nonMSIH': '#83c7e9',
}

plt.figure(figsize=(2.6,2.5))
unique_labels = np.unique(string_labels)
for label in cancer_colors.keys():
    indices = sample_df[sample_df["PCR"] == label].index.tolist()
    plt.scatter(patient_features_2d[indices,0], patient_features_2d[indices, 1], label=label,s=20, alpha=0.1, color=cancer_colors[label])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()
plt.close()

reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(features)

plt.figure(figsize=(2.6,2.5))
unique_labels = np.unique(string_labels)
for label in cancer_colors.keys():
    indices = sample_df[sample_df["PCR"] == label].index.tolist()
    plt.scatter(embedding[indices,0], embedding[indices, 1], label=label,s=20, alpha=0.1, color=cancer_colors[label])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()
plt.close()
# %%