import json
import sys

from os import path
from pile import load_model
from train import parse_input_json
from wefe.metrics import RNSB, WEAT
from wefe.query import Query
from wefe.word_embedding_model import WordEmbeddingModel

def fetch_queries(input_queries):
    """Method to fetch queries from data/raw/queries.json file.
    
    Keyword arguments:
    input_queries -- query, format defined by WEFE
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

def evaluate_metrics(input_file, query_data, model, model_name):
    """ Based on the given query, a word embedding model gets evaluated using WEAT, RNSB metrics
    
    Keyword arguments:
    model -- instance of a word embedding model, model
    model_name -- name of a word embedding model, str
    """
    #Dictionary for storing metric scores for a model and different query combination
    #intermediate_res = {}
    for query in query_data:
        # loop here for getting multiple queries in the wefe Query format
        new_query = Query(target_sets = query['target_sets'],
                          attribute_sets = query['attribute_sets'],
                          target_sets_names = query['target_sets_names'],
                          attribute_sets_names = query['attribute_sets_names']
                          )

        we_model = WordEmbeddingModel(model.wv, model_name)

        res={}

        res['model_id'] = model_name
        res['query'] = query

        for metric in input_file['metrics']:
        
            if metric == 'WEAT':
                weat_res = WEAT().run_query(new_query, we_model, lost_vocabulary_threshold=0.5, calculate_p_value=True, p_value_iterations=15000)
                res['weat_result'] = weat_res['weat']
                print(we_model.name, weat_res['query_name'], 'WEAT', weat_res['weat'])

            elif metric == 'RNSB':
                rnsb_res = RNSB().run_query(new_query, we_model, lost_vocabulary_threshold=0.5, calculate_p_value=True, p_value_iterations=15000)
                res['rnsb_result'] = rnsb_res['rnsb']
                print(we_model.name, rnsb_res['query_name'], 'RNSB', rnsb_res['rnsb'])
        
    return res

def main():
    """ Program to evaluate embeddings using social bias metrics(WEAT, RNSB)"""
    
    #Store results in a dictionary
    results = {}
    results['model_id'] = []
    results['weat_results'] = []
    results['rnsb_results'] = []
    results['query'] = []

    #Parse input file
    input_file_data = parse_input_json(sys.argv[1])

    #Fetch query from input file
    query_data = fetch_queries(input_file_data['queries'])
    
    results_file_path = input_file_data['results_file_path'] 
    
    if 'heatmap_plot' in input_file_data:
        results['heatmap_plot'] = input_file_data['heatmap_plot']
    
    if 'radar_plot' in input_file_data:
        results['radar_plot'] = input_file_data['radar_plot']

    for i in input_file_data['models']:
        #fetch model file path
        model_file_path = i['model_save_path']
        
        #get unique model name/id
        model_name = i['id']
        model_exists_flag = path.exists(model_file_path)
        if model_exists_flag == True:
            print('Model found')
            model = load_model(model_file_path)
            #Evaluate embeddings using bias metrics WEAT and RNSB
            model_res = evaluate_metrics(input_file_data, query_data, model, model_name)

            results['model_id'].append(model_res['model_id'])
            results['weat_results'].append(model_res['weat_result'])
            results['rnsb_results'].append(model_res['rnsb_result'])
            results['query'].append(model_res['query'])
        else:
            print('** Model not found under : ',model_file_path,'**')

    if 'results_file_path' in input_file_data:
        with open(results_file_path, 'w') as fp:
            json.dump(results, fp, indent=4, sort_keys=True)
        print('Results saved under: ', results_file_path)


if __name__ == "__main__":
    main()
