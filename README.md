# SocialBias
<hr />
<!-- Identifiers-->

Measuring social bias is critical for better understanding and measuring unfairness in NLP/ML models.

`SocialBias` provides pipeline structure for evaluating custom trained word embeddings models (`Word2Vec`, `Glove` etc) on social bias metrics (such as `WEAT`, `RNSB` etc). We can easily mix and match different datasets, word embedding models, queries and social bias metrics. This framework helps to conduct experiments to investigate the influence of different embedding and model parameters on bias metrics.
### :bookmark_tabs: Get Started

You can find all information needed to get started with `SocialBias` here.

### :wrench: Installation
 All the required packages to run the scripts  can be installed using the provided <a rel="pipfile" href="https://git.cs.uni-paderborn.de/vinay/socialbias/-/blob/main/Pipfile">`Pipfile`</a> and `pipenv` commands,
```shell
$ pipenv shell
$ pipenv install
```

###  :paperclip: Usage
`Socialbias` framework consists of scripts to train word embedding models on custom datasets, evaluate these word embeddings using social bias metrics and these results are then visualized using different plots. With these scripts used in tandem, we achieve a pipeline structure for training, evaluation and visualization of word embeddings.

`Experiment file` is used to specify the model parameters, bias metric parameters, result file paths and visualization plot flags(`heatmap_plot` and `radar_plot`). More details are provided in the next section. Experiment files are placed under `/socialbias/data/experiments/` folder. File path of an experiment file is then set as the 'INPUT_FILE_PATH' variable in the <a rel="pipelinefile" href="https://git.cs.uni-paderborn.de/vinay/socialbias/-/blob/main/src/run_pipeline.sh">`run_pipeline.sh`</a>. Pipeline is run as below,

```shell
$ bash run_pipeline.sh
```

Pipeline initiates a series of processing scripts that are placed in `/socialbias/src/` folder. `train.py` takes an experiment file as input, trains word embedding models and saves them in `/socialbias/models/` folder. `metrics.py` loads the word embedding models and evaluates them using social bias metrics specified in the experiment file and stores their results in a file(<a rel="resultsfile" href="https://git.cs.uni-paderborn.de/vinay/socialbias/-/blob/main/data/results/case1/res-case1-w2v-pile-workers-1_8_32.json">example results file</a>). `plot_results` takes the results file as the input and saves the visualizations in `.png` format. Execution of individual scripts is as shown below,

```shell
$ python bias.py ../data/experiments/case1/case1_input.json

$ python metrics.py ../data/experiments/case1/case1_input.json

$ python plot_results.py ../data/results/case1/w2v-pile-workers.json
```

`cosine_distances.py` computes cosine distances between word embeddings of a list of `50` randomly chosen words. The same experiment file is passed as input,

```shell
$ python cosine_distances.py ../data/experiments/case1/case1_input.json
```

### :page_facing_up: Experiment File
Parameters and inputs required for an experiment are defined in a `JSON` structure. Multiple word embedding models(`word2vec` or `fasttext`) with different parameters, social bias metrics(`WEAT` and `RNSB`) and their parameters, flags for visualizing results, folder paths to save models and results can be specified in the experiment file. Every experiment is uniquely identified by their `case_id`.

Here is an example experiment file,

```json

   {
    "models": [
        {
            "id": "w2v-pile-workers-8",
            "model_name": "word2vec",
            "PYTHONHASHSEED" : 8,
            "dataset": "pile",
            "model_save_path": "../models/case1/w2v-pile-workers-8",
            "word2vec" : {
            "vector_size": 300,
            "workers": 8
            }
        },
        {
            "id": "w2v-pile-workers-32",
            "model_name": "word2vec",
            "PYTHONHASHSEED" : 8,
            "dataset": "pile",
            "model_save_path": "../models/case1/w2v-pile-workers-32",
            "word2vec" : {
            "vector_size": 300,
            "workers": 32
            }
        },
        {
            "id": "w2v-pile-workers-1",
            "model_name": "word2vec",
            "PYTHONHASHSEED" : 8,
            "dataset": "pile",
            "model_save_path": "../models/case1/w2v-pile-workers-1",
            "word2vec" : {
            "vector_size": 300,
            "workers": 1
            }
        }
    ],
    "metrics" : ["WEAT", "RNSB"],
    "queries" : [
        {
            "queries_path_name" : "../data/raw/queries.json",
            "target_sets_names" : ["gender", "female", "male"],
            "attribute_sets_names" : ["family", "career"]
        }
    ],
    "case_id" : "w2v-pile-workers",
    "results_file_path" : "../data/results/case1/w2v-pile-workers.json",
    "heatmap_plot" : true,
    "radar_plot" : true
    }
```

Also allows to reference queries from a standard list of queries(<a href="https://git.cs.uni-paderborn.de/vinay/socialbias/-/blob/main/data/raw/queries.json">`queries`</a>) as shown above.

### :scroll: License
<table>
  <tr>
    <td><a rel="license" href="https://opensource.org/licenses/MIT"><img alt="MIT License" style="border-width:0" width="160px" src="http://ipo.opencircularity.info/wp-content/uploads/2019/03/MIT-License-transparent.png" /></a></td>
    <td>
    This project is licensed under the <a rel="license" href="https://git.cs.uni-paderborn.de/vinay/socialbias/-/blob/main/LICENSE">MIT License.</a></td>
  </tr>
</table>




