#!/usr/bin/env python

# This script lets you serialize the process of,
# Training your choice of word embeddings(word2vec or fasttext) on specific datasets(cc_news, Pile),
# Evaluating these embeddings using social bias metrics(WEAT, RNSB) and 
# visualize the metric scores (in the form of heatmap plots or radar diagrams)  
# based on the parameters in the experiments(input) file

INPUT_FILE_PATH="../data/experiments/case2/case2_input.json" 

#parse results_file_path from the experiments file
RESULTS_FILE_PATH=$(jq .results_file_path "${INPUT_FILE_PATH}") 

#code to remove double quotes in the results file path
RESULTS_FILE_PATH=`sed -e 's/^"//' -e 's/"$//' <<<"$RESULTS_FILE_PATH"` 

#-------------------TRAINING-------------------------

python train.py "${INPUT_FILE_PATH}" && echo "TRAINING COMPLETE"

#--------------METRIC EVALUATION----------------------

python metrics.py "${INPUT_FILE_PATH}" && echo "EMBEDDINGS EVALUATED"

#--------------SCORES VISUALIZATION-------------------

python plot_results.py "${RESULTS_FILE_PATH}"  
