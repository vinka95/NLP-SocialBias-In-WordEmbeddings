import pandas as pd
import gensim
import json
import gc
import tempfile
import sys

from os import path
from datasets import load_dataset
from gensim.utils import simple_preprocess


def word2vec_model(text, parameters):
    """Instantiates a word2vec model and builds vocabulary for the model.

    Keyword arguments:
    text -- dataset, set of tokenized sentences
    workers -- no of workers used to train word2vec model

    Return:
    model -- instance of a word2vec model
    """
    # Instantiates gensim implementation of a Word2Vec model
    model = gensim.models.Word2Vec(**parameters)

    # builds vocabulary for the model
    print("\nBuilding vocabulary")
    model.build_vocab(text, progress_per=10000)
    return model

def fasttext_model(text, workers):
    """Instantiates a fasttext model and builds vocabulary for the model.

    Keyword arguments:
    text -- dataset, set of tokenized sentences
    workers -- no of workers used to train

    Return:
    model -- instance of a fasttext model
    """
    # Instantiates gensim implementation of a Fasttext model
    model = gensim.models.FastText()

    print("\nBuilding vocabulary")
    # build the vocabulary
    model.build_vocab(corpus_file=corpus_file)

    return model

def save_model(model, file_path):
    """Saves a model in the given file path

    Keyword arguments:
    model -- instance of a word embedding model
    filepath -- path to save model, str
    """

    model.save(file_path)
    print("Model Saved")
    save_model_flag = True

    return save_model_flag

def load_model(model_filepath):
    """Loads a saved model from the given model path

    Keyword arguments
    model_filepath -- path to load model from, str
    """
    print("\nLoading model from : " + model_filepath)
    new_model = gensim.models.Word2Vec.load(model_filepath)

    return new_model
