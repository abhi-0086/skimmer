import nltk
import string 
# Download NLTK resources if not already downloaded 
nltk.download('punkt') 
def split_sentences(abstract): 
    """Split abstract into sentences.""" 
    sentences = nltk.sent_tokenize(abstract) 
    return sentences 

def tokenize_sentence(sentence): 
    """Tokenize sentence into words.""" 
    tokens = nltk.word_tokenize(sentence) 
    return tokens 

def tokenize_sentences(sentences): 
    """Tokenize multiple sentences into words.""" 
    tokenized_sentences = [tokenize_sentence(sentence) for sentence in sentences] 
    return tokenized_sentences 

def char_vectorize_token(token): 
    """Convert a token to a character vector.""" 
    char_vector = list(token) 
    return char_vector 

def char_vectorize_sentence(sentence): 
    """Convert sentence to character vectors.""" 
    #char_vectors = [char_vectorize_token(token) for token in tokenize_sentence(sentence)] 
    #return char_vectors 
    return " ".join(list(sentence))

def char_vectorize_sentences(sentences): 
    """Convert multiple sentences to character vectors.""" 
    char_vectorized_sentences = [char_vectorize_sentence(sentence) for sentence in sentences] 
    return char_vectorized_sentences