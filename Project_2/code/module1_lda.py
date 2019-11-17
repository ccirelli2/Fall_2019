# FALL 2019 - GSU IMAGE PROCESSING - PROJECT 2 ---------------------------------------------

# IMPORT PYTHON PACKAGES -------------------------------
import re
import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime
import string

#Nltk
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# pyLDAvis
import pyLDAvis
import pyLDAvis.gensim

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


# PREPROCESS TEXT -------------------------------------
def preprocess_txt(txt):
    # Start timer
    print('Starting clearning process')
    start = datetime.now()
    # Remove Punctuation
    rm_punct            = [sentence.translate({ord(i): None for i in string.punctuation}).lower()                              for sentence in txt]
    # Tokenize sentences
    tokenize_sentences  = [sentence.split(' ') for sentence in rm_punct]
    # Remove Stopwords
    rm_stop_words_ints  = [
            [token for token in sentence if token not in stop_words and token.isalpha() is True]
             for sentence in tokenize_sentences]
    # Lemmetize Tokens
    token_lemmas        = [[lemmatizer.lemmatize(token) for token in sentence]
                            for sentence in rm_stop_words_ints]
    # End Timer
    end = datetime.now()
    # Print Time to run
    print('Time to run preprocessing => {}\n'.format(end-start))
    # Return
    return token_lemmas



# TRAIN LDA MODEL ---------------------------------------
def lda_train_model(corpus, id2word, num_topics, random_state, 
        update_every, chunksize, passes, return_topics=False):
    print('\nTraining LDA model')
    start_time = datetime.now()
    lda_model = gensim.models.ldamodel.LdaModel(    corpus      = corpus,
                                                    id2word     = id2word,
                                                    num_topics  = num_topics,
                                                    random_state= random_state,
                                                    update_every= update_every,
                                                    chunksize   = chunksize,
                                                    passes      = passes,
                                                    alpha       = 'auto',
                                                    per_word_topics=True)
    end_time = datetime.now()
    print('Model training finished.  Time to completion => {}'.format(end_time-start_time))

    # Return Topics
    if return_topics == True:
        print('\nReturning Topics')
        pprint(lda_model.print_topics())


    # Return trained model
    return lda_model


# OBTAIN OPTIMAL NUMBER OF TOPICS ----------------------------

def get_perplexity_score(lda_model, corpus):
    lda_complexity = lda_model.log_perplexity(corpus)
    return lda_complexity


def get_coherence_score_num_topics( min_num_topics, max_num_topics, corpus, id2word, 
                                    random_state, update_every, chunksize, passes, 
                                    texts, coherence_measure, plot=False, print_topics=False, 
                                    show_topic_bubbles=False):
    '''
    Purpose:        Calculate the coherence score from the lda model over n-topics
    min/max_topics  Minimum and maximum number of topics to test
    corpus:         List of unique words in all text
    id2word:        dictionary of id/count pairs for all texts
    random_state:   seed to start random selection
    update_every:   see docs
    texts:          cleaned version of original text (list of lists containing tokens)
    Output:         List of tuples, (topic_num, coherence_score)
    '''
    # List - Coherence Score (num_topics, score)
    num_topics_l        = []
    log_perplexity_l    = [] 
    coherence_scores_l  = []

    # Train LDA on N Topics
    for num_topics in range(min_num_topics, max_num_topics):
        lda_model_output = lda_train_model(corpus, id2word, num_topics=num_topics,
            random_state=random_state, update_every=update_every, chunksize=chunksize, 
            passes=passes)
        
        # Print Topics
        if print_topics == True:
            lda_model_output.print_topics()

        if show_topic_bubbles == True:
            vis = pyLDAvis.gensim.prepare(lda_model_output, corpus, id2word)
            dir_output = r'/home/ccirelli2/Desktop/GSU/Fall_2019/Project_2/output'
            pyLDAvis.save_html(vis, dir_output + '/' + 'vis.html')
            print('Topic bubble graph saved to {}'.format(dir_output))
        # Generate Coherence Score
        log_perplexity_score= lda_model_output.log_perplexity(corpus)
        coherence_model_lda = CoherenceModel(model=lda_model_output, texts = texts, 
                              dictionary=id2word, coherence=coherence_measure)
        coherence_score     = coherence_model_lda.get_coherence()
        
        # Append Num Topics & Score to List
        num_topics_l.append(num_topics)
        log_perplexity_l.append(log_perplexity_score)
        coherence_scores_l.append(coherence_score)

        print('Score score generated for topic num => {}'.format(num_topics))
        print('Log Perplexity Score => {}'.format(log_perplexity_score))
        print('Coherence Score      => {}'.format(coherence_score))

    # Create Dataframe
    df = pd.DataFrame({})
    df['num_topics']            = num_topics_l
    df['coherence_scores']      = coherence_scores_l
    df['log_perplexity_score']  = log_perplexity_l

    # Plot Scores
    if plot_scores == True:
        
        # Coherence 
        df_plot_coh     = df['coherence_scores'].plot(kind='line', 
                            use_index=False, grid=True, legend=True, fontsize=15)
        plt.title('COHERENCE SCORE')
        
        plt.show()
        
        # Perplexity
        df_plot_perp    = df['log_perplexity_score'].plot(kind='line', 
                            use_index=False, grid=True, legend=True, fontsize=15)
        plt.title('PERPLEXITY SCORE')
        plt.show()
    

    # Return List of Scores to User
    return df










