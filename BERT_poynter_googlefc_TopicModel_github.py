# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:57:59 2021

@author: Ida Anthonj Nissen
"""

#BERT topic modeling to identify the topics automatically present in two datasets (COVID misinformation stories from Google Fact-check Explorer and Poynter institute)

#%% Import packages
import pandas as pd
import numpy as np

#%% Import data from poynter and google
df_claims_poynter = pd.read_csv (r'PATH\Poynter\list_claimtitles_poynter_FebSept.csv')
df_claims_googlefc = pd.read_csv (r'PATH\Google_FC\list_claimtitles_googlefc_FebSept.csv')

print(df_claims_poynter.head(5))

list_claim_poynter = df_claims_poynter['0'].tolist()
list_claim_googlefc = df_claims_googlefc['0'].tolist()


#%% Topic modeling

#Article: Topic modeling with BERT: BERTopic
#https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

#Embed the sentences
# BERT sentence transformers
# Follow https://github.com/UKPLab/sentence-transformers

#download pre-trained model
from sentence_transformers import SentenceTransformer

#Choose best pre-trained model from https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('distilbert-base-nli-mean-tokens')  #model used by BERTopic

embeddings1 = model.encode(list_claim_poynter)
embeddings2 = model.encode(list_claim_googlefc)

#Reduce dimensionality with UMAP
#_neighbours and n_components can be changed
import umap
umap_embeddings1 = umap.UMAP(n_neighbors=30, n_components=25, metric='cosine').fit_transform(embeddings1)
umap_embeddings2 = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine').fit_transform(embeddings2)

#Cluster sentences with HDBSAN
import hdbscan
cluster1 = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings1)
cluster2 = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings2)


#%% Poynter
#visualize resulting clusters
import matplotlib.pyplot as plt

# Prepare data
umap_data1 = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings1)
result = pd.DataFrame(umap_data1, columns=['x', 'y'])
result['labels'] = cluster1.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
#random colors for colormap
vals = np.linspace(0,1,max(clustered.labels)+1) #comment out to keep the same colors
np.random.shuffle(vals)     #comment out to keep the same colors
cmap = plt.cm.colors.ListedColormap(plt.cm.hsv(vals))
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=60)
#plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=60, cmap=cmap)
#plt.colorbar()
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks([0,10])
cbar.set_ticklabels([1,11])
plt.axis('off')
  
#export figure
fig.savefig('PATH\BERT_topicmodel\Fig_poynter_topics_scatter_300dpi.png', dpi = 300)




#Find important words in each cluster using a class-based TF-IDF
#First create a single document of all sentences for each cluster
docs_df = pd.DataFrame(list_claim_poynter, columns=["Doc"])
docs_df['Topic'] = cluster1.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

#Then apply c-TF-IDF
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(list_claim_poynter))

#Take top 20 words based on their c-TF-IDF scores
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
#View how many sentences there are in the largest topics
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(20)
#NB: -1 are outliers, they were not assigned to a topic

for i in range(len(topic_sizes)):
    if topic_sizes.iloc[i][0] != -1:   #if first condition is false second condition is not tested
        vis = pd.DataFrame(top_n_words[topic_sizes.iloc[i][0]][:15])
        print(vis.head(15)) 



#View the top words for the top topics:
columns = []
for i in range(1,len(topic_sizes)):
    columns.append(top_n_words[topic_sizes.iloc[i][0]][:15])

#Export    
df_top_n_words = pd.DataFrame(columns)
df_top_n_words = df_top_n_words.transpose()

df_top_n_words.to_csv('PATH\BERT_topicmodel\df_poynter_topwords.csv', index=False, encoding='utf-8-sig', sep=';')
df_top_n_words.to_excel('PATH\BERT_topicmodel\df_poynter_topwords.xlsx', index=False, encoding='utf-8-sig')

docs_df.to_csv('PATH\BERT_topicmodel\df_poynter_docs2topic.csv', index=False, encoding='utf-8-sig', sep=';')
docs_df.to_excel('PATH\BERT_topicmodel\df_poynter_docs2topic.xlsx', index=False, encoding='utf-8-sig')

docs_per_topic.to_csv('PATH\BERT_topicmodel\df_poynter_docs_per_topic.csv', index=False, encoding='utf-8-sig', sep=';')
docs_per_topic.to_excel('PATH\BERT_topicmodel\df_poynter_docs_per_topic.xlsx', index=False, encoding='utf-8-sig')

topic_sizes.to_csv('PATH\BERT_topicmodel\df_poynter_topic_sizes.csv', index=False, encoding='utf-8-sig', sep=';')
topic_sizes.to_excel('PATH\BERT_topicmodel\df_poynter_topic_sizes.xlsx', index=False, encoding='utf-8-sig')





#%% Google FC

#Reduce dimensionality with UMAP
#_neighbours and n_components can be changed
umap_embeddings5 = umap.UMAP(n_neighbors=20, n_components=10, metric='cosine').fit_transform(embeddings2)

#Cluster sentences with HDBSAN
cluster5 = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings5)


#visualize resulting clusters
# Prepare data
umap_data5 = umap.UMAP(n_neighbors=20, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings2)
result = pd.DataFrame(umap_data5, columns=['x', 'y'])
result['labels'] = cluster5.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
#random colors for colormap
vals = np.linspace(0,1,max(clustered.labels)+1) #comment out to keep the same colors
np.random.shuffle(vals) #comment out to keep the same colors
cmap = plt.cm.colors.ListedColormap(plt.cm.hsv(vals))   #comment out to keep the same colors
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=60)
#plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=10, cmap='hsv_r')
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=60, cmap=cmap)
#plt.colorbar()
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks([0,12])
cbar.set_ticklabels([1,13])
plt.axis('off')
  
#export figure
fig.savefig('PATH\BERT_topicmodel\Fig_googlefc_topics_scatter.png', dpi = 300)


#Find important words in each cluster using a class-based TF-IDF
#First create a single document of all sentences for each cluster
docs_df = pd.DataFrame(list_claim_googlefc, columns=["Doc"])
docs_df['Topic'] = cluster5.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

#Then apply c-TF-IDF
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(list_claim_googlefc))

#Take top 20 words based on their c-TF-IDF scores
top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
#View how many sentences there are in the largest topics
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(20)
#NB: -1 are outliers, they were not assigned to a topic
#View the top words for the top topics:
top_n_words[topic_sizes.iloc[1][0]][:10]
top_n_words[topic_sizes.iloc[2][0]][:10]
top_n_words[topic_sizes.iloc[3][0]][:10]


for i in range(len(topic_sizes)):
    if topic_sizes.iloc[i][0] != -1:   #if first condition is false second condition is not tested
        vis = pd.DataFrame(top_n_words[topic_sizes.iloc[i][0]][:15])
        print(vis.head(15)) 


topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
#View the top words for the top topics:
columns = []
for i in range(1,len(topic_sizes)):
    columns.append(top_n_words[topic_sizes.iloc[i][0]][:15])

#Export    
df_top_n_words = pd.DataFrame(columns)
df_top_n_words = df_top_n_words.transpose()

df_top_n_words.to_csv('PATH\BERT_topicmodel\df_googlefc_topwords.csv', index=False, encoding='utf-8-sig', sep=';')
df_top_n_words.to_excel('PATH\BERT_topicmodel\df_googlefc_topwords.xlsx', index=False, encoding='utf-8-sig')

docs_df.to_csv('PATH\BERT_topicmodel\df_googlefc_docs2topic.csv', index=False, encoding='utf-8-sig', sep=';')
docs_df.to_excel('PATH\BERT_topicmodel\df_googlefc_docs2topic.xlsx', index=False, encoding='utf-8-sig')

docs_per_topic.to_csv('PATH\BERT_topicmodel\df_googlefc_docs_per_topic.csv', index=False, encoding='utf-8-sig', sep=';')
docs_per_topic.to_excel('PATH\BERT_topicmodel\df_googlefc_docs_per_topic.xlsx', index=False, encoding='utf-8-sig')

topic_sizes.to_csv('PATH\BERT_topicmodel\df_googlefc_topic_sizes.csv', index=False, encoding='utf-8-sig', sep=';')
topic_sizes.to_excel('PATH\BERT_topicmodel\df_googlefc_topic_sizes.xlsx', index=False, encoding='utf-8-sig')


  
#%% Loop to find optimal parameters

#loop
for neigh in [10, 15, 20, 25, 30]:
    for comp in [5, 10, 15, 20, 25, 30, 50]:
        for min_cl in [5, 10, 15, 20, 25, 30, 50]:
            
            #Found to be optimal for Poynter dataset:
            #neigh = 30
            #comp = 25
            #min_cl = 50
            
            #Found to be optimal for Google dataset:
            #neigh = 20
            #comp = 20
            #min_cl = 10

            #_neighbours and n_components can be changed
            umap_embeddings2 = umap.UMAP(n_neighbors=neigh, n_components=comp, metric='cosine').fit_transform(embeddings2)
            
            #Cluster sentences with HDBSAN
            cluster2 = hdbscan.HDBSCAN(min_cluster_size=min_cl, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings2)
            
            #Find important words in each cluster using a class-based TF-IDF
            #First create a single document of all sentences for each cluster
            #docs_df = pd.DataFrame(list_claim_googlefc, columns=["Doc"])
            docs_df = pd.DataFrame(list_claim_googlefc, columns=["Doc"])
            docs_df['Topic'] = cluster2.labels_
            docs_df['Doc_ID'] = range(len(docs_df))
            docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
            
            #Then apply c-TF-IDF
            tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(docs_df))
            
            #Take top 20 words based on their c-TF-IDF scores
            top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
            
            #Output: topic sizes and top words
            topic_sizes = extract_topic_sizes(docs_df)
            print('neigh ', neigh, 'comp ', comp, 'min_cl ', min_cl)
            print(topic_sizes.head(10))
            #View the top words for the top topics:
            for i in range(0,10):
                if len(topic_sizes) > i and topic_sizes.iloc[i][0] != -1:   #if first condition is false second condition is not tested
                    print('topic idx ', i, 'parameters', neigh, comp, min_cl)
                    vis = pd.DataFrame(top_n_words[topic_sizes.iloc[i][0]][:10])
                    print(vis.head(10)) 
                else:
                    print('i ', i, 'Not executed', neigh, comp, min_cl)
            
            
            
            vis = pd.DataFrame(top_n_words[topic_sizes.iloc[1][0]][:10])
            print(vis.head(10))
            vis = pd.DataFrame(top_n_words[topic_sizes.iloc[2][0]][:10])
            print(vis.head(10))
            vis = pd.DataFrame(top_n_words[topic_sizes.iloc[3][0]][:10])
            print(vis.head(10))
            

