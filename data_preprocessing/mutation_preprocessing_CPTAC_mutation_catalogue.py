#%%
import pickle
import pandas as pd

path = '...'
tcga_maf = pickle.load(open(path+'cptac_crc_maf_table_sbs_mutation_catalogue.pkl', 'rb')) # or indel

CR = ["Missense_Mutation","Silent","Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins"]
tcga_maf = tcga_maf[tcga_maf["Variant_Classification"].isin(CR)]

tcga_dfs = []
patients = tcga_maf['case_id'].unique().tolist()
for patient in patients:
    df = tcga_maf[tcga_maf["case_id"]==patient]
    tcga_dfs.append(df)

def find_microhomology_bool(deletion_seq, prefix, suffix):
    deletion_seq = deletion_seq.replace("-", "")  # remove gaps from deletion sequence
    max_length = min(len(deletion_seq), len(prefix), len(suffix))
    longest_beginning = ""
    longest_end = ""
    
    # check for longest beginning match with suffix
    for i in range(1, max_length + 1):
        if deletion_seq[:i] == suffix[:i]:
            longest_beginning = deletion_seq[:i]
        else:
            break

    # check for longest end match with prefix
    for i in range(1, max_length + 1):
        if deletion_seq[-i:] == prefix[-i:]:
            longest_end = deletion_seq[-i:]
        else:
            break 
    # ensure the match is partial and not the entire deletion sequence
    if (len(deletion_seq) == len(longest_beginning)) or (len(deletion_seq) == len(longest_end)):
        return False
    else:
        if (longest_beginning and len(longest_beginning) < len(deletion_seq)) or (longest_end and len(longest_end) < len(deletion_seq)):
            if longest_beginning:
                print(f"Longest microhomology at the beginning: {longest_beginning} (Length: {len(longest_beginning)})")
            if longest_end:
                print(f"Longest microhomology at the end: {longest_end} (Length: {len(longest_end)})")
            return True
        else:
            return False
    
def find_microhomology_int(deletion_seq, prefix, suffix):
    deletion_seq = deletion_seq.replace("-", "")  # remove gaps from deletion sequence
    max_length = min(len(deletion_seq), len(prefix), len(suffix))
    longest_beginning = ""
    longest_end = ""
    
    # check for longest beginning match with suffix
    for i in range(1, max_length + 1):
        if deletion_seq[:i] == suffix[:i]:
            longest_beginning = deletion_seq[:i]
        else:
            break

    # check for longest end match with prefix
    for i in range(1, max_length + 1):
        if deletion_seq[-i:] == prefix[-i:]:
            longest_end = deletion_seq[-i:]
        else:
            break
    return max(len(longest_beginning),len(longest_end))

def classify_indels_detailed(five,ref,alt,three):
    indel_mutation_dict = {"1DelC1":0,"1DelC2":0,"1DelC3":0,"1DelC4":0,"1DelC5":0,"1DelC6":0,"1DelT1":0,"1DelT2":0,"1DelT3":0,"1DelT4":0,"1DelT5":0,"1DelT6":0,"1InsC0":0,"1InsC1":0,"1InsC2":0,"1InsC3":0,"1InsC4":0,"1InsC5":0,"1InsT0":0,"1InsT1":0,"1InsT2":0,"1InsT3":0,"1InsT4":0,"1InsT5":0,"2DelRep1":0,"2DelRep2":0,"2DelRep3":0,"2DelRep4":0,"2DelRep5":0,"2DelRep6":0,"3DelRep1":0,"3DelRep2":0,"3DelRep3":0,"3DelRep4":0,"3DelRep5":0,"3DelRep6":0,"4DelRep1":0,"4DelRep2":0,"4DelRep3":0,"4DelRep4":0,"4DelRep5":0,"4DelRep6":0,"5DelRep1":0,"5DelRep2":0,"5DelRep3":0,"5DelRep4":0,"5DelRep5":0,"5DelRep6":0,"2InsRep0":0,"2InsRep1":0,"2InsRep2":0,"2InsRep3":0,"2InsRep4":0,"2InsRep5":0,"3InsRep0":0,"3InsRep1":0,"3InsRep2":0,"3InsRep3":0,"3InsRep4":0,"3InsRep5+":0,"4InsRep0":0,"4InsRep1":0,"4InsRep2":0,"4InsRep3":0,"4InsRep4":0,"4InsRep5":0,"5InsRep0":0,"5InsRep1":0,"5InsRep2":0,"5InsRep3":0,"5InsRep4":0,"5InsRep5":0,"2MH1":0,"3MH1":0,"3MH2":0,"4MH1":0,"4MH2":0,"4MH3":0,"5+MH1":0,"5+MH2":0,"5+MH3":0,"5+MH4":0,"5+MH5":0}
    num_mut = len(five)
    for i in range(len(five)):
        row_five = five[i]
        row_three = three[i]
        row_ref = ref[i] 
        row_alt = alt[i] 
        if set(row_alt) == {'-'}: 
            if row_ref[1] == '-':
                if row_ref[0] == "G" or "C":
                    classi = "1DelC"
                    lenC = len([char for char in row_three if char == 'C' and row_three.startswith('C' * (row_three.index(char) + 1))])
                    lenG = len([char for char in row_three if char == 'G' and row_three.startswith('G' * (row_three.index(char) + 1))])
                    lenadd = max(lenC,lenG) + 1
                    if lenadd > 6 :
                        lenadd = 6
                    indel_mutation_dict[classi+str(lenadd)] +=1
                elif row_ref[0] == "T" or "A":
                    classi = "1DelT"
                    lenT = len([char for char in row_three if char == 'T' and row_three.startswith('T' * (row_three.index(char) + 1))])
                    lenA = len([char for char in row_three if char == 'A' and row_three.startswith('A' * (row_three.index(char) + 1))])
                    lenadd = max(lenT,lenA) + 1
                    if lenadd > 6 :
                        lenadd = 6
                    indel_mutation_dict[classi+str(lenadd)] +=1
            elif row_ref[2] == '-':              
                if find_microhomology_bool(row_ref,row_five,row_three):
                    indel_mutation_dict["2MH1"] +=1
                else: 
                    classi = "2DelRep"
                    main_string = row_three
                    sub_string = row_ref[:2]
                    count = sum(main_string[i:i+len(sub_string)] == sub_string for i in range(0, len(main_string) - len(sub_string) + 1, len(sub_string)))
                    if count > 5 :
                        count = 5
                    indel_mutation_dict[classi+str(count+1)] +=1
            elif row_ref[3] == '-':              
                if find_microhomology_bool(row_ref,row_five,row_three):
                    indel_mutation_dict["3MH"+str(find_microhomology_int(row_ref,row_five,row_three))] +=1
                else:
                    classi = "3DelRep"
                    main_string = row_three
                    sub_string = row_ref[:3]
                    count = sum(main_string[i:i+len(sub_string)] == sub_string for i in range(0, len(main_string) - len(sub_string) + 1, len(sub_string)))
                    if count > 5:
                        count = 5
                    indel_mutation_dict[classi+str(count+1)] +=1
            elif row_ref[4] == '-':              
                if find_microhomology_bool(row_ref,row_five,row_three):
                    indel_mutation_dict["4MH"+str(find_microhomology_int(row_ref,row_five,row_three))] +=1
                else:
                    classi = "4DelRep"
                    main_string = row_three
                    sub_string = row_ref[:4]
                    count = sum(main_string[i:i+len(sub_string)] == sub_string for i in range(0, len(main_string) - len(sub_string) + 1, len(sub_string)))
                    if count > 5 :
                        count = 5
                    indel_mutation_dict[classi+str(count+1)] +=1
            elif any(row_ref[j] == '-' for j in range(5, len(row_ref))) or '-' not in row_ref:   
                if find_microhomology_bool(row_ref,row_five,row_three):
                    if find_microhomology_int(row_ref,row_five,row_three)>5:
                        indel_mutation_dict["5+MH5"] +=1
                    else:
                        indel_mutation_dict["5+MH"+str(find_microhomology_int(row_ref,row_five,row_three))] +=1
                else:
                    classi = "5DelRep"
                    main_string = row_three
                    sub_string = row_ref[:5]
                    count = sum(main_string[i:i+len(sub_string)] == sub_string for i in range(0, len(main_string) - len(sub_string) + 1, len(sub_string)))
                    if count > 5 :
                        count = 5
                    indel_mutation_dict[classi+str(count+1)] +=1

        elif set(row_ref) == {'-'}: 
            if row_alt[1] == '-': 
                if row_alt[0] == "G" or "C":
                    classi = "1InsC"
                    lenC = len([char for char in row_three if char == 'C' and row_three.startswith('C' * (row_three.index(char) + 1))])
                    lenG = len([char for char in row_three if char == 'G' and row_three.startswith('G' * (row_three.index(char) + 1))])
                    lenadd = max(lenC,lenG)
                    if lenadd > 5:
                        lenadd = 5
                    indel_mutation_dict[classi+str(lenadd)] +=1
                elif row_alt[0] == "T" or "A":
                    classi = "1InsT"
                    lenT = len([char for char in row_three if char == 'T' and row_three.startswith('T' * (row_three.index(char) + 1))])
                    lenA = len([char for char in row_three if char == 'A' and row_three.startswith('A' * (row_three.index(char) + 1))])
                    lenadd = max(lenT,lenA)
                    if lenadd > 5:
                        lenadd = 5
                    indel_mutation_dict[classi+str(lenadd)] +=1
            elif row_alt[2] == '-':
                    classi = "2InsRep"
                    main_string = row_three
                    sub_string = row_ref[:2]
                    count = sum(main_string[i:i+len(sub_string)] == sub_string for i in range(0, len(main_string) - len(sub_string) + 1, len(sub_string)))
                    if count > 5 :
                        count = 5
                    indel_mutation_dict[classi+str(count)] +=1
            elif row_alt[3] == '-': 
                    classi = "3InsRep"
                    main_string = row_three
                    sub_string = row_ref[:3]
                    count = sum(main_string[i:i+len(sub_string)] == sub_string for i in range(0, len(main_string) - len(sub_string) + 1, len(sub_string)))
                    if count > 5 :
                        count = 5
                    indel_mutation_dict[classi+str(count)] +=1
            elif row_alt[4] == '-': 
                    classi = "4InsRep"
                    main_string = row_three
                    sub_string = row_ref[:4]
                    count = sum(main_string[i:i+len(sub_string)] == sub_string for i in range(0, len(main_string) - len(sub_string) + 1, len(sub_string)))
                    if count > 5 :
                        count = 5
                    indel_mutation_dict[classi+str(count)] +=1
            elif any(row_alt[j] == '-' for j in range(5, len(row_alt))) or '-' not in row_alt: 
                    classi = "5InsRep"
                    main_string = row_three
                    sub_string = row_ref[:5]
                    count = sum(main_string[i:i+len(sub_string)] == sub_string for i in range(0, len(main_string) - len(sub_string) + 1, len(sub_string)))
                    if count > 5 :
                        count = 5
                    indel_mutation_dict[classi+str(count)] +=1
    for key in indel_mutation_dict.keys():
        indel_mutation_dict[key] =  indel_mutation_dict[key] / (num_mut)
    return indel_mutation_dict

def sbs_dict():
    sbs_dic = {}
    for mut_base in ["C>A","C>G","C>T","T>A","T>C","T>G"]:
        for five_prime in ["A","C","T","G"]:
            for three_prime in ["A","C","T","G"]:
                key = five_prime+mut_base+three_prime
                sbs_dic[key] = 0
    return sbs_dic

def classify_mutations(five,ref,alt,three):
    valid_bases = 'ACGT'
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    sbs_mutation_dict = sbs_dict()
    for i in range(len(five)):
        five_prime = five[i]
        ref_base = ref[i]
        alt_base = alt[i]
        three_prime = three[i]
        if ref_base in 'GA' and all(base in valid_bases for base in [five_prime, ref_base, alt_base, three_prime]):
            rev_five_prime = complement[three_prime]
            rev_three_prime = complement[five_prime]
            rev_ref_base = complement[ref_base]
            rev_alt_base = complement[alt_base]
            mutation_class = rev_three_prime + rev_ref_base + '>' + rev_alt_base + rev_five_prime
        elif all(base in valid_bases for base in [five_prime, ref_base, alt_base, three_prime]):
            mutation_class = five_prime + ref_base + '>' + alt_base + three_prime 
        else:
            print("other mutation")  
            print(five_prime+alt_base+">"+ref_base+three_prime)
        sbs_mutation_dict[mutation_class] +=1
    num_mut = len(five)
    for key in sbs_mutation_dict.keys():
        sbs_mutation_dict[key] =  sbs_mutation_dict[key] / (num_mut)
    return sbs_mutation_dict

#%%
# use either this block or the next one dependent on which mutation type you use
dict_list = []
for dataframe in tcga_dfs:
    five_l = dataframe["five_p"].tolist()
    three_l = dataframe["three_p"].tolist()
    ref_l = [x[0] for x in dataframe["Ref"].tolist()]
    alt_l = [x[0] for x in dataframe["Alt"].tolist()]
    sbs_catalogue = classify_mutations(five_l,ref_l,alt_l,three_l)
    sbs_catalogue["case_id"] = dataframe.case_id.unique()[0]
    dict_list.append(sbs_catalogue)
sbs_catalogue_df = pd.DataFrame(dict_list)
sbs_catalogue_df

sbs_catalogue_df.to_csv(path+"cptac_crc_sbs_mutation_catalogues_norm.csv",index=False)
# %%
dict_list = []
for dataframe in tcga_dfs:
    dataframe = dataframe.dropna(subset=['Ref', 'Alt','five_p','three_p']) 
    five_l = dataframe["five_p"].tolist()
    three_l = dataframe["three_p"].tolist()
    ref_l = [x for x in dataframe["Ref"].tolist()]
    alt_l = [x for x in dataframe["Alt"].tolist()]
    id_catalogue = classify_indels_detailed(five_l,ref_l,alt_l,three_l)
    id_catalogue["case_id"] = dataframe.case_id.unique()[0]
    dict_list.append(id_catalogue)
id_catalogue_df = pd.DataFrame(dict_list)

id_catalogue_df.to_csv(path+"cptac_crc_id_mutation_catalogues_norm.csv",index=False)