import pandas as pd
import gensim
import json
import gc
import tempfile
import sys

from models import word2vec_model, fasttext_model, save_model, load_model
from cc_news import dataset_cc_news, data_preprocess
from pile import pile_train

from os import path

# Define Pile dataset paths
# pile_base_path = path.join("file:///", "mnt", "ceph", "storage", "corpora", "corpora-thirdparty", "the-pile")
pile_base_path = path.join("file:///", "mnt", "ceph", "storage", "data-tmp", "current", "mspl", "data", "the-eye.eu", "public", "AI", "pile")
val_data_path = path.join(pile_base_path, "val.jsonl.zst")
test_data_path = path.join(pile_base_path, "test.jsonl.zst")
train_data_paths = [path.join(pile_base_path, "train", f"{str(n).zfill(2)}.jsonl.zst") for n in range(0, 30)]

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

    # returns JSON object as a dictionary
    data = json.load(f)

    f.close()

    return data

def fetch_queries(input_queries):
    """Method to fetch queries from data/raw/queries.json file.
    
    Keyword arguments:
    input_queries -- query

    Return:
    query sets -- parsed target and attribute sets either from the input experiments file or from the queries.json source file
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
            #convert text list to dataframe for easier processing
            text_df = pd.DataFrame(text)
            print("\nPreprocessing dataset text" )
            p_text = data_preprocess(text_df)
            
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
