import pandas as pd
import json
import gensim
import gc
import sys

from models import word2vec_model
from os import path
from gensim.utils import simple_preprocess

# Define Pile dataset paths
# pile_base_path = path.join("file:///", "mnt", "ceph", "storage", "corpora", "corpora-thirdparty", "the-pile")
pile_base_path = path.join("file:///", "mnt", "ceph", "storage", "data-tmp", "current", "mspl", "data", "the-eye.eu", "public", "AI", "pile")
val_data_path = path.join(pile_base_path, "val.jsonl.zst")
test_data_path = path.join(pile_base_path, "test.jsonl.zst")
train_data_paths = [path.join(pile_base_path, "train", f"{str(n).zfill(2)}.jsonl.zst") for n in range(0, 30)]

# Data set selection, comment line to exclude dataset from Pile 
data_selection = [
    'OpenWebText2'
    'PubMed Abstracts',
    'StackExchange',
    # 'Github', # Currently ignoring because we don't want the code
    'Enron Emails',
    'FreeLaw',
    'USPTO Backgrounds',
    'Pile-CC',
    'Wikipedia (en)',
    'Books3',
    'PubMed Central',
    'HackerNews',
    'Gutenberg (PG-19)'
    # 'DM Mathematics', # Currently ignoring because we don't want math formulas
    'NIH ExPorter',
    'ArXiv',
    'BookCorpus2',
    'OpenSubtitles',
    'YoutubeSubtitles',
    'Ubuntu IRC',
    # 'EuroParl', # Currently ignoring because we'll focus on English text for now
    'PhilPapers'
]

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
    #add code for loading fasttext model

    print("\nLoading model from : " + model_filepath)
    new_model = gensim.models.Word2Vec.load(model_filepath)

    return new_model

def data_preprocess(df):
    """Method to preprocess the text by applying a simple_preprocess function provided by pandas.
       Given a text, it gets tokenized and removes punctuations and special characters

    Keyword arguments:
    text -- dataset, text

    Return:
    proc_text -- processed dataset, text
    """
    #apply simple preprocessing(tokenization, removal of punctuations and special characters) on text
    proc_text = df['text'].apply(simple_preprocess)
    return proc_text

def clear_variables():
    """ Clears local variables """
    del data_chunk, filtered_data, df_text, processed_text, model
    gc.collect()

def pile_train(data_paths, parameters, save_file_path):
    """Trains a word embedding model on Pile dataset in batches.
    
    Keyword arguments:
    data_paths -- path to pile files, str
    parameters -- word embedding model parameters, json
    save_file_path -- file path to store model, str

    Return:
    model_flag -- model trained if flag is set to True 
    """
    model_flag = False
    batch_num = 1
    # Iterating through training paths
    for file_path in train_data_paths[7:]:
        # Read 'zstd' compressed file and iterate through the file in batches of given chunksize 
        with pd.read_json(file_path, lines=True, chunksize=1000000, compression='zstd') as reader:
            reader
            for data_chunk in reader:
                print("---------------")
                print('Batch: ',batch_num, '\n File: ', file_path)

                # Transform set name column to make it easier to work with
                data_chunk["meta_str"] = data_chunk["meta"].apply(lambda x: x["pile_set_name"])
                print('Chunk size: ', len(data_chunk))

                # Only select the data we are interested in for now
                filtered_data = data_chunk[data_chunk["meta_str"].isin(data_selection)][["text", "meta_str"]]
                print('Chunk size after filtering: ',len(filtered_data))

                # Convert filtered data to pandas dataframe for easier processing
                df_text = pd.DataFrame(filtered_data['text']).reset_index()

                print("\nPreprocessing dataset text" )
                processed_text = data_preprocess(df_text)

                print(model_flag)
                if model_flag == False:
                    save_file_flag = path.exists(save_file_path)
                    if save_file_flag == False:
                        # If a model does not exist, instantiate a word2vec and build vocabulary
                        model = word2vec_model(processed_text, parameters)
                        print(model)
                    else:
                        model = load_model(save_file_path)
                        model.build_vocab(processed_text, update = True)
                        print(model)
                else:
                    model = load_model(save_file_path)
                    model.build_vocab(processed_text, update = True)
                    print(model)

                # Training model
                model.train(processed_text, total_examples= model.corpus_count, epochs= model.epochs)

                # Saving model
                model_flag = save_model(model, save_file_path)
                batch_num = batch_num + 1

                # Clear variables and free memory
                # clear_variables()
                del data_chunk, filtered_data, df_text, processed_text, model
                gc.collect()

    return model_flag
