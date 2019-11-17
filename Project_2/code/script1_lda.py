# FALL 2019 - GSU IMAGE PROCESSING - PROJECT 2 ---------------------------------------------
'''
Gensim Documentation:   https://radimrehurek.com/gensim/auto_examples/index.html
gensim.models.phrases:  Looks like the purpose is to work with the collocation of words 'from a 
                        stream of sentences'.
                        It actually combines words w/ an underscore. ex: tree_house, and uses
                        frequency of the collocation of the words to determine if they should be
                        combined or not. 
                        url:    https://radimrehurek.com/gensim/models/
                                phrases.html#gensim.models.phrases.Phrases
                        
min_count:              The minimum collocation of two words
threshold:              The minimum score for a bigram to be taken into account. 
Scorer:                 See docs https://radimrehurek.com/gensim/models/
                        phrases.html#gensim.models.phrases.npmi_scorer

corpora.dictionary      Maps words and their integer ids
        `               gensim creates a unique id for each word in the document. mapping is
                        [word_id, word_frequency].  
                        Note:   that it appears that the word frequency is for the document and 
                                not the entire corpus. 
                        Note:   if you want to return the word associated with an id 
                        just pass the id to the dictionary object, ex:  id2word[0]
                        Url:    https://radimrehurek.com/gensim/corpora/dictionary.html

LDA Model               Input:  Two main inputs are the corpora.Dictionary(sentences) and
                                corpus 'id2word.doc2bow'

                        Documentation:  https://radimrehurek.com/gensim/models/ldamodel.html
                        Random:         I think because the initial assignment is random
                        Update_every:   Number of documents to be iterated through for each update. 
                                        0 for batch learning and > 1 for online iterative learning. 
                        chunksize:      Number of documents to be used in each training chunk
                        passes:         Number of passes through the corpus during training
                                        per_word_topics If true, the model compute a list of 
                                        topics in order of importane. 
Model Output            Produces n different topics where each topic is a combination of keywords
                        and each word contributes a certain weight to the topic. These are topics
                        over the entire corpous and not individual documents. 

Topic Coherence         Interpreting key words is subjective.  Model is not gauranteed to be 
                        inpretable.  Score is used to measure how well the topics are extracted. 
                        Optimal Number of Subjects:  Train many LDA models with different 
                        numbers of topics and pick the one that gives the highest coherence
                        score. 
                        
                        Paper:  http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
                        article 'evaluating topic modeling':    
                        https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/

                        defined:    Measures the score a single topic by measuring the degree of
                                    semantic similarity between high scoring words in the topic. 
                                    These measurements help distinguish between topics that are
                                    semantically interpretable topics and topics that are 
                                    artifacts of statistical inferenes. 
                        coherence:  A set of facts is said to be coherent if they support each 
                                    other.   


Topic Perplexity        It captures how surprised a model is of new data it has not seen before
                        and is measured as the normalized log-likelihood of a held-out test
                        set.

                        Url:    https://towardsdatascience.com/evaluate-topic-model-in-python-
                                latent-dirichlet-allocation-lda-7d57484bb5d0


'''

# IMPORT PYTHON PACKAGES --------------------------------------------------
import re
import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime
import string
import random 

#Nltk
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


# Spacy
import spacy

# Plotting
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
import matplotlib.pyplot as plt

# Stop Words
from nltk.corpus import stopwords
stop_words = [word for word in stopwords.words('english')]

# Project Modules
import module1_lda as m1

# LOAD DATA ----------------------------------------------------------------
afile = r'M_fund.xlsx'
dir_  = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/data'
path2file = dir_ + '/' + afile
df = pd.read_excel(path2file).dropna()

# CREATE SAMPLE SET --------------------------------------------------------
random_num  = random.randint(5000,6000)
df          = df.iloc[: random_num]


# INSPECT DATA -------------------------------------------------------------
print('Dimensions   => {}'.format(df.shape))
print('Column Names => {}'.format(df.columns))

# ISOLATE PRINCIPAL STRATEGIES (ROWS W/ TEXT) ------------------------------
df_strategies = df['principal_strategies']

# PREPROCESS TEXT ----------------------------------------------------------
clean_txt = m1.preprocess_txt(df_strategies)

# CREATE A DICTIONARY AND CORPUS FOR MODELING ------------------------------
id2word   = Dictionary(clean_txt)
corpus    = [id2word.doc2bow(sentence) for sentence in clean_txt]


# BUILD TOPIC MODEL / GEN COHERENCE & PERPLEXITY SCORES --------------------
'''The lda model is built into the coherence score function
'''
def coherence_scores():
    coherence_scores = m1.get_coherence_score_num_topics(
        min_num_topics=10, max_num_topics = 11, corpus=corpus, id2word=id2word, random_state=100, 
            update_every=1, chunksize=100, passes=10, texts=clean_txt, coherence_measure='c_v', 
            plot=True, print_topics=False, show_topic_bubbles=True)
        
    print(coherence_scores)

coherence_scores()

# VIZUALIZE MODEL ----------------------------------------------------
num_topics      = [1,2,3,4,5,6,7,8,9,10,11,12]
coherence_score = [0.312, 0.342, 0.358, 0.374, 0.366, 0.40, 0.408, 0.397, 0.380, 0.408, 0.415]
perplexity_score= [-6.44, -6.34, -6.27, -6.227, -6.21, -6.19, -6.17, -6.18, -6.21, -6.25, -6.34]

# Plot Coherence Score
'''
plt.plot(coherence_score)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Number of Topics', fontsize=14)
plt.title('LDA MODEL - COHERENCE SCORE', fontsize=16)
plt.show()
plt.plot(perplexity_score)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Number of Topics', fontsize=14)
plt.title('LDA MODEL - PERPLEXITY SCORE', fontsize=16)
plt.show()
'''













