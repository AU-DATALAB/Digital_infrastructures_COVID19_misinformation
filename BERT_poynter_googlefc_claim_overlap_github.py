# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:57:59 2021

@author: Ida Anthonj Nissen
"""

#BERT for overlap of titles of the misinformation claims between the Poynter and Google FC Expl. database

#%% Import packages
import pandas as pd
import numpy as np
import tensorflow as tf

#%% Import google and poynter claim list

df_claims_poynter = pd.read_csv (r'PATH\Poynter\list_claimtitles_poynter_FebSept.csv')
df_claims_googlefc = pd.read_csv (r'PATH\Google_FC\list_claimtitles_googlefc_FebSept.csv')

print(df_claims_poynter.head(5))

list_claim_poynter = df_claims_poynter['0'].tolist()
list_claim_googlefc = df_claims_googlefc['0'].tolist()


#%% BERT sentence transformers
# Follow https://github.com/UKPLab/sentence-transformers

#download pre-trained model
from sentence_transformers import SentenceTransformer, util

#Semantic textual similarity
#Choose best pre-trained model from https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('stsb-roberta-large')

#  Embed a list of sentences
sentences1 = list_claim_poynter#[0:3]
sentences2 = list_claim_googlefc#[0:3]

embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    
#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
#convert into values
cos_val = []
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        cos_val.append(cosine_scores[i][j].item())


dict_comp = []
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        #dict_comp.append({'index': [i, j], 'poynter': sentences1[i], 'googlefc': sentences2[j], 'score': cosine_scores[i][j], 'score_val': cosine_scores[i][j].item()})
        dict_comp.append({'index': [i, j], 'poynter': sentences1[i], 'googlefc': sentences2[j]})

dict2 = {'score_val': cos_val}
dict_comp.update(dict2)

dict_comp['score_val'] = cos_val
dict_comp['score_val'].append(cos_val)
print(dict_comp[0])
print(dict_comp[0:10])
        

#get value out of tensor object:
#cosine_scores[1,1].item()        
#dict_comp[0]['score'].item()

#Sort scores in decreasing order
dict_comp = sorted(dict_comp, key=lambda x: x['score_val'], reverse=True)

#Show top matching claims
for pair in dict_comp[0:10]:
    i, j = pair['index']
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], pair['score']))
       



