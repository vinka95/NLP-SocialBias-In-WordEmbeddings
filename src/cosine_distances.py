import gensim
import os, sys
import random
import time
import json

from train import parse_input_json
from itertools import combinations
from scipy import spatial

models_list = []

def cosine_distances(word_list):
    """For a word from the word list, computes the cosine distance between the word vectors of models (say model1 to model2,            model1 to model3, model1 to model4, model1 to model5, model2 to model3, model2 to model4, ...)

    Keyword arguments:
    word_list -- list of words for which cosine distances are computed

    Return:
    distances -- list of tuples, (word, average of cosine distances)
    """
    distances = []
    word_dist_tuple = ()
    for word in word_list:
        dist_sum = 0
        num = 0
        word_dict = {}
        word_dict['word_name'] = word
        for i in combinations(models_list, 2):
            num += 1
            indiv_dist = spatial.distance.cosine(i[0][0].wv[word], i[1][0].wv[word])
            dist_sum += indiv_dist
            # code for storing individual model distances(model 1 -- model 2, model 1 -- model 3 etc)
            # word_dict : {
            #     word_name: man
            #     model1-model2: 0.23,
            #     model1-model3: 0.23,
            #     avg_distance: 0.23
            # }
            word_dict[str(i[0][1])+'--'+str(i[1][1])] = indiv_dist

        # word_dist_tuple = (word, dist_sum/num)
        word_dict['average'] = dist_sum/num
        distances.append(word_dict)

    return distances

def save_results(data, save_file_path):
    """Method to save the results into a .json file.

    Keyword arguments:
    data -- tuples of words and their average cosine distances
    save_plot_path -- path to save the embedding cosine distances
    """

    out_file = open(save_file_path, "w")
    json.dump(data, out_file, indent=4)

    print('\nResults saved under : ', save_file_path)

def main():
    model_file_paths = []
    random_word_list = []
    cosine_distances_folder_path = '../data/results/cosine_distances/'

    dir_path = sys.argv[1]
    
    #Parse input file
    input_file_data = parse_input_json(sys.argv[1])
    
    #compute date for file_name
    timestr = time.strftime("-%Y-%m-%d_%H-%M")
    
    #get results file path
    results_file_path = input_file_data['results_file_path']
    left, delim, right = results_file_path.rpartition('/')
    results_file_path = cosine_distances_folder_path + right[:-5] + '-cosine_differences' + timestr + '.json'

    # get all model file_paths
    for model in input_file_data['models']:
        model_file_paths.append(model['model_save_path'])
    
    # load and store instances of models in a list 
    for model_file in model_file_paths:
        model = gensim.models.Word2Vec.load(model_file)
        models_list.append((model, model_file))

    #generate a list of random words
    random_word_list = random.choices(model.wv.index_to_key, k=50)
    
    print('\nComputing cosine distances')
    results = cosine_distances(random_word_list)

    print('\nSaving results')
    save_results(results, results_file_path)

if __name__ == "__main__":
    main()
