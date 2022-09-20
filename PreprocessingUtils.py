import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer


'''Term Frequencey converts a given corpus into a frequency representation (based on the frequency of 
a word)'''
def initialise_term_frequency_vectorizer(data,ngram = 1,stopwords = False):
    if(stopwords):
      vectorizer_tf = CountVectorizer(ngram_range=(ngram,ngram),stop_words= 'english')
    else:
      vectorizer_tf = CountVectorizer(ngram_range=(ngram,ngram))

    vectorizer_tf.fit(data)
    X = vectorizer_tf.transform(data)
    return X, vectorizer_tf


'''Converts word -> ID dictionary to ID -> word dictionary'''
def get_id2word(vocabulary):
    id2word = {}
    for key in vocabulary.keys():
      id2word[vocabulary[key]] = key
    return id2word


'''Term frequency-inverse document frequency. This is a statistic that is based on the frequency of 
a word in the corpus but it also provides a numerical representation of how important a word is for statistical analysis'''
def initialise_tfidf_vectorizer(data,ngram = 1,stopwords = False):
    if(stopwords):
       vectorizer_tfidf = TfidfVectorizer(ngram_range=(ngram,ngram),stop_words= 'english')
    else:
       vectorizer_tfidf = TfidfVectorizer(ngram_range=(ngram,ngram))
   
    vectorizer_tfidf.fit(data)
    X = vectorizer_tfidf.transform(data)
    return X, vectorizer_tfidf


'''Word 2 vec model, converts vocabulary into vector representation
 https://jonathan-hui.medium.com/nlp-word-embedding-glove-5e7f523999f6
 https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
 Two methods to create word embeddings:
 Continous bag of words - Given context words predict target word
 Skip Gram given the target predict context words
 '''
def initialise_word_to_vec_model(data,vocab_size,min_freq_count=2,skip_gram=0,workers=2,num_of_epochs=5):
    word_vec_model = Word2Vec(data,min_count = min_freq_count,vector_size = vocab_size, sg = skip_gram,workers=workers)
    vocabulary = word_vec_model.build_vocab(data)
    X = word_vec_model.train(data,totoal_examples=word_vec_model.corpus_count,epochs=num_of_epochs)
    return X, vocabulary, word_vec_model

'''Tokenize text using keras'''
def tokenize_data(data):
  # create the dict.
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(data)
   return tokenizer



'''
Get embeddings matrix from corpus using embedding file
'''
def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index) + 1
      
    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim))
  
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
  
    return embedding_matrix_vocab
  
  
"""
In GloVe, we measure the similarity of the hidden factors
between words to predict their co-occurrence count. Viewed from 
this perspective, we do not predict the co-occurrence words only.
 We want to create vector representations that can predict their
  co-occurrence counts in the corpus also.
"""

'''
Word embedding encodes words. But it does not account for its word context
we will look at vector representations for sentences that can be used for many NLP tasks. BERT is used by Google in its search and good for many NLP tasks. '''
def test():

   
   
    test_data = fetch_20newsgroups(subset='train')

    #Testing term frequency 
    # X, vectorizer_tf  = initialise_term_frequency_vectorizer(test_data['data'],stopwords=True)
    # id2word = get_id2word(vectorizer_tf.vocabulary_)
    # token_counts = X.sum(axis=0)
    # list_token_counts = token_counts.tolist()[0]
    # sorted_index = np.argsort(list_token_counts)[::-1]
    # print("Highest occuring word:%s - count:%s" % (id2word[sorted_index[0]], list_token_counts[sorted_index[0]]))

    #Testing term frequency-inverse document frequency
    # X1, vectorizer_tfidf = initialise_tfidf_vectorizer(test_data['data'],stopwords=True)
    # id2word = get_id2word(vectorizer_tfidf.vocabulary_)
    # token_counts = X1.sum(axis=0)
    # list_token_counts = token_counts.tolist()[0]
    # sorted_index = np.argsort(list_token_counts)[::-1]
    # print("Highest occuring word:%s - count:%s" % (id2word[sorted_index[0]], list_token_counts[sorted_index[0]]))
    
    #Testing word to vec
    # X, vocab, model = initialise_word_to_vec_model(test_data['data'],len(test_data['data']),num_of_epochs=1)
    # print(X)


    #Testing tensorflow tokenizer
    #tokenizer = tokenize_data(test_data['data'][0:1])
    #print(tokenizer.index_word)
    #print(tokenizer.word_index)
    
if __name__ == "__main__":
    test()

