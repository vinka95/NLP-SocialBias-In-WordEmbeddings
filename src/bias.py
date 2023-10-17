import pandas as pd
import gensim
import json
import gc
import tempfile
import sys

from wefe.metrics import RNSB, WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel


from os import path
from datasets import load_dataset
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

def parse_input_json(file_path):
    """Method to parse a given input file.

    Keyword arguments:
    file_path -- path to file containing the input parameters

    Return:
    data -- parsed data from the input file
    """
    print('Input file path: ',file_path)
    # Opening JSON file
    f = open(file_path)
    
    # print(f)

    # returns JSON object as a dictionary
    data = json.load(f)

    f.close()

    return data

def fetch_queries(input_queries):
    """Method to fetch queries from data/raw/queries.json file.
    
    Keyword arguments:
    input_queries -- query
    """
    query_sets = []
    for input_query in input_queries:
        query = {}
        if 'queries_path_name' in input_query:
            # When path to a queries file is given

            # declaring local variables for target_sets and attribute_sets
            # query = {}
            target_sets = []
            attribute_sets = []

            # Opening JSON file
            query_file = open(input_query['queries_path_name'])

            # returns queries JSON object as a dictionary
            raw_query_data = json.load(query_file)
            
            target_sets.append(raw_query_data['target_sets'][input_query['target_sets_names'][0]][input_query['target_sets_names'][1]]['set'])
            target_sets.append(raw_query_data['target_sets'][input_query['target_sets_names'][0]][input_query['target_sets_names'][2]]['set'])
            
            # print(raw_query_data['attribute_sets'][input_query['attribute_sets_names'][0]]['set'])
            attribute_sets.append(raw_query_data['attribute_sets'][input_query['attribute_sets_names'][0]]['set'])
            attribute_sets.append(raw_query_data['attribute_sets'][input_query['attribute_sets_names'][1]]['set'])

            # fills up the query dictionary
            query['target_sets'] = target_sets
            query['attribute_sets'] = attribute_sets
            query['target_sets_names'] = [input_query['target_sets_names'][1], input_query['target_sets_names'][2]]
            query['attribute_sets_names'] = input_query['attribute_sets_names']

            # Closing file
            query_file.close()

        else:
            # when query target and attribute sets are explicitly given
            query['target_sets'] = input_query['target_sets']
            query['attribute_sets'] = input_query['attribute_sets']
            query['target_sets_names'] = input_query['target_sets_names']
            query['attribute_sets_names'] = input_query['attribute_sets_names']
                                             
        query_sets.append(query)

    return query_sets

def dataset_cc_news():
    """Method to get cc_news dataset that is provided by Huggingface.

    Return:
    text -- cc_news dataset
    """
    #Load cc_news dataset from datasets provided by Huggingface
    dataset = load_dataset('cc_news', split = 'train')

    #Keep only the text from the dataset
    text = dataset['text']
    return text

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

def evaluate_metrics(model, query_data, model_name):
    """ Based on the given query, a word embedding model gets evaluated using WEAT, RNSB metrics
    
    Keyword arguments:
    model -- instance of a word embedding model, model
    model_name -- name of a word embedding model, str
    """
    new_query = Query(target_sets = query_data[0]['target_sets'],
                               attribute_sets = query_data[0]['attribute_sets'],
                               target_sets_names = query_data[0]['target_sets_names'],
                               attribute_sets_names = query_data[0]['attribute_sets_names']
                               )


    we_model = WordEmbeddingModel(model.wv, model_name)
    
    weat_res = WEAT().run_query(new_query, we_model, lost_vocabulary_threshold=0.5, calculate_p_value=True, p_value_iterations=15000)
    
    print(we_model.name, weat_res['query_name'], 'WEAT', weat_res['weat'])
    
    rnsb_res = RNSB().run_query(new_query, we_model, calculate_p_value=True, lost_vocabulary_threshold=0.5, p_value_iterations=15000)
    
    print(we_model.name, rnsb_res['query_name'], 'RNSB', rnsb_res['rnsb'])
    

def save_model(model, file_path):
    """Saves a model in the given file path
    
    Keyword arguments:
    model -- instance of a word embedding model
    filepath -- path to save model, str
    """
    
    # file_path = "../models/pile/word2vec_pile" 
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
<<<<<<< HEAD
    for file_path in train_data_paths[2:]:
        print(file_path)
=======
    for file_path in train_data_paths[29:]:
>>>>>>> [GEN] Adds modularised scripts to train, evaluate and vizualize word embeddings
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
                
                # print(parameters)
                
                # Apply simple preprocessing(tokenization, removal of punctuations and special characters) on text
                # measure how much time?? How to make apply function on multiple cores?
                processed_text = df_text['text'].apply(simple_preprocess)
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
<<<<<<< HEAD
                    model.build_vocab(processed_text, update=True)
=======
                    model.build_vocab(processed_text, update = True)
>>>>>>> [GEN] Adds modularised scripts to train, evaluate and vizualize word embeddings
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

def data_preprocess(text):
    """Method to preprocess the text by applying a simple_preprocess function provided by pandas.
       Given a text, it gets tokenized and removes punctuations and special characters

    Keyword arguments:
    text -- dataset, text

    Return:
    proc_text -- processed dataset, text
    """
    #convert text list to dataframe for easier processing
    df = pd.DataFrame(text)

    #apply simple preprocessing(tokenization, removal of punctuations and special characters) on text
    proc_text = df[0].apply(simple_preprocess)
    return proc_text

def main():
    """Method to parse input file, load dataset, train a model, evaluate metrics and store results.
    """
    #variable definitions
    results = []
    model_flag = False

    # Parse input file
    input_data = parse_input_json(sys.argv[1])

    # Fetching queries from input file ../data/experiments/case1_input2.json
    query_data = fetch_queries(input_data['queries']) 
    
    # Below code to be used after updating python to 3.10
    # ---
    # match input_data['model_name']:
    #     case ["word2vec"]:
    #         model_parameters = input_data['models']['word2vec']
    #     case _:
    #         model_parameters = None
    # ---
    
    for model in input_data['models']:
        # Fetch model specific parameters
        if model['model_name'] == 'word2vec':
            model_parameters = model['word2vec']
        elif model['model_name'] == 'fasttext':
            model_parameters = model['fasttext']
        
        model_save_path = model['model_save_path']
        
        # Fetch specific Dataset to train the model on
        if model['dataset'] == 'cc_news':
            print("---------------")
            print("Loading dataset: ", model['dataset'] )
            text = dataset_cc_news()
            print("\nPreprocessing dataset text" )
            p_text = data_preprocess(text)
            
            print("---------------")
            print("Word Embedding Model: " + model['model_name'])
            print("Model parameters: ")
            print(model_parameters)
            
            # Instantiate word2vec and build vocabulary
            model = word2vec_model(processed_text, parameters)
            
            #Train model             
            model.train(p_text, total_examples=temp_model.corpus_count, epochs=temp_model.epochs)
            
            # Saving model
            save_flag = save_model(model, model_save_path)
            
        elif model['dataset'] == 'pile':
            print("---------------")
            print("Loading dataset: ", model['dataset'])
            print("---------------")
            print("Word Embedding Model: " + model['model_name'] + " (Training in batches)")
            print("Model parameters: ")
            print(model_parameters)

            model_trained = pile_train(train_data_paths, model_parameters, model_save_path)

if __name__ == "__main__":
    main()
