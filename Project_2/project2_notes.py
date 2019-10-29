# PROJECT DELIVERABLES ---------------------------------------------------------
'''
1.  LDA model:  best output (log-likelihood, coherence store, bubble visualization)
2.  Dynamic topic model (topics change over time)
3.  LDA2VEC:    how does this compare to the LDA model? Improvement methods such as using
                word2vec pre-trained data
4.  LSTM generate new texts.  refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/


Deliverables:
1.  Code:       notebook and html
2.  Excel file: topic-key words distributions output
3.  Presentation slides

Python Libraries (noted in presentation):
1. LSA:     from sklearn.feature_extraction.text import TfidVectorizer
            from sklearn.decomposition import TruncatedSVD
                
2. LDA:     https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
*continue page 49
'''


# FURTHER RESEARCH -------------------------------------------------------------
'''
1.) RNN         Find a good video to explain how they are structured and work. 


'''


# SLIDES - WORD REPRESENTATION & FEATURE EXTRACTION ---------------------------- 
'''
Tf-idf Matrix   
    tf          term frequecy or count matrices of each word in each document
    idf         inverse document frequency * 
                log( #documents / (#documents containing word W)

Ngrams          consecutive word sequences of a given length

Distributional hypothesis of language:
                - states that the meaning of a word can be inferred from the context
                  in which it is used.
                - "a word is characterized by the company it keeps" was popularized by 
                  First (1957).  
                - Distributional Hypothesis is the basis for Statistical
                  Semantics. 
                - Linguistic Semantics: 
                    "linguistic items with similar distributions have
                    similar meanings".  Idea is to use linear algebra
                    to define distributiona/semantic similarity in terms of vector similarity
                    https://en.wikipedia.org/wiki/Distributional_semantics
                - By observing the co-occurance patterns of words across a large body of
                  text we can discover that the context in whcih burger occur are quite 
                  similar to those in which pizza occurs, less similar to those in which
                  ice cream occurs and very different from chair. 

Two Methods Learn Categories:
                - Cluster based models:  assign a word to a group. 
                - Embedding based methods:  represent words as vectors such that
                                            similar words have similar distributions

Singular Value Decomposition:
                - A special case of this is called LSA or Latent Semantic Analysis
                - NN models = Skip-grams and CBOW. 

Latent Semantic Analysis:
                - One of the foundational techniques in topic modeling. 
                - Core idea:  take a matrix of documents and terms and decompose it into a 
                  a separate document-topic matrix and topic-term matrix. 
                - Matrix:   each row represents a document and each column a word. 
                -           simplest version is to use a raw count of the number of times
                            the j-th word appeard in the i-th document. 
                            That said, we should replace count w/ Tf-idf
                            **Since the term-document matrix is sparse, we do truncated
                            SVD, which factorizes any matrix M into the product of 3
                            separate matrices, M = U X S X V, where S is a diagonal 
                            matrix of singular values of M.  
                            **Truncated SVD reduces the dimensionality by selecting
                            the t largest singular values.
                - For analyzing relationships between a set of documents and the terms
                  they contain.  LSA assumes that words that are close in meaning will 
                  occur in similar pieces of text. 
                - A matrix is created with rows representing unique words and columns 
                  paragraphs.  Then SVM is used to reduce the number of ros while 
                  preserving the similarity structure amonth columns. 
                  Paragraphs are then compared by taking the cosine of the angle between
                  the two vectors formed by any two columns. Values of 1 represent very 
                  similar paragraphs while 0 represent dissimilar paragraphs. 

Brown Clustering:       
                - Cluster owrds based on whcih words precede or follow them. 
                - These word clusters can be turned into a kind of vector

Skip-gram:
                - Given a set of sentences (also called corpus) the model loops on the 
                  words of each sentence and either tries to use the current word to predict
                  its neighbors (its context). 

CBOW:
                - Refers to continous bag of words.  Tries to predict the current words
                  by taking the words around it as input variables. The limit on the number
                  of words in each context is determined by a paramater called "window size". 
                - reference:  https://towardsdatascience.com/word2vec-skip-gram-model
                              -part-1-intuition-78614e4d6e0b

Cosine Similarity:
                - Measures the cosine of the angle between vectors. 
                - The cosine of 0 degrees is 1, and it is less than 1 for any angle in 
                  the interval (0, pi] radians.  It is thus a judgement of orientation and
                  not magnitude. 
                - Two vectors with the same orientation have a cosine similar to 1. 
                - Two vectors disimilar to one another have an orientation at 90 degrees. 
                - Two vectors diametrically opposed have a similarity of -1. 
                - Ref: https://www.mathopenref.com/triggraphcosine.html
                - Ref:  https://blog.exploratory.io/demystifying-text-analytics-finding-similar-documents-with-cosine-similarity-e7b9e5b8e515
                - Calculation:
                        Sigma (u * v) / sqrt(Sigma(u^2)) _ sqrt(sigma(v^2))


Topic Modeling:
                - Topic is the main idea discussed in a text data. 
                - Could be topic of a sentence, paragram, an article or entire corpus. 



LDA:            - Stands for Latent Dirichlet Allocation
                - Each topic is a distribution over words
                - Each document is a mixture of corpus-wide topics
                - Each word is drawn from one of those topics. 
                - *Trying to figure out which words tend to co-occur and will cluster them
                  into distinct topics. 
                - **Implementation - Gensim / Tipic Modeling w/ Scikit Learn


'''


