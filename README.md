# SolvingHardAnalogyQuestions

This repo provides the source code & data of our paper titled "Solving Hard Analogy Questions with Relation Embedding Chains", presented at EMNLP 2023 main conference.

## Citation
If you use this work in your research, please cite our paper:

```
@InProceedings{kumarn8emnlp,
  author    = {Nitesh Kumar and Steven Schockaert},
  title     = {Solving Hard Analogy Questions with Relation Embedding Chains},
  year      = {2023},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
}
```

## Setup Instructions

### Installation:

```commandline
conda create -n analogies python=3.10.11
conda activate analogies
pip install relbert unidecode gensim git+https://github.com/yuce/pyswip@ab3a36d#egg=pyswip lmdb redis matplotlib xgboost
conda install -c conda-forge python-rocksdb
```

### Download resources:
*  Download the "checkpoint.zip" file containing trained models and additional files from [this link](https://cf-my.sharepoint.com/:u:/g/personal/kumarn8_cardiff_ac_uk/EWuKxOgsCj1NmRU4r3gCbTQB3nUewT3S9UtHh4VssLuP5A?e=n4PBsn)
*  Unzip it and move the unziped `checkpoint` folder to the appropriate location:

```commandline
unzip checkpoint.zip
sudo mkdir -p /scratchtest/c.scmnk4/elexir/
sudo chmod 777 /scratchtest/c.scmnk4/elexir/
mv checkpoint /scratchtest/c.scmnk4/elexir/resources
```

### Download source code:
```commandline
git clone https://github.com/niteshroyal/SolvingHardAnalogyQuestions.git
mkdir reasoning_with_vectors
mv SolvingHardAnalogyQuestions/* reasoning_with_vectors/
mv reasoning_with_vectors SolvingHardAnalogyQuestions/
cd SolvingHardAnalogyQuestions/

export PYTHONPATH="${PYTHONPATH}:/your_path_to/SolvingHardAnalogyQuestions"
```

## Reproducing the results

Move to your working directory:

```commandline
cd SolvingHardAnalogyQuestions/
```

The configuration file is `reasoning_with_vectors/conf/exper_conf.py`. Set the location of the logging folder in that file:

```
logging_folder = "/your_path_to/SolvingHardAnalogyQuestions/reasoning_with_vectors/logs"
```

### Running the Direct Approach

To evaluate the Direct approach on the analogy datasets, first store all RelBERT embeddings required for evaluation in the LMDB Store:
```commandline
python reasoning_with_vectors/core/preprocessing.py
```

This may take a lot of time. To tracke the progress see the logs files in `logs` folder. This step is not mandatory, but executing it will speed up the evaluation process since RelBERT will not be called each time for relation embedding.

Now, to evaluate the Direct approach, first run:

```commandline
python reasoning_with_vectors/experiments/evaluation.py
```

This will create a pickle file in the `/scratchtest/c.scmnk4/elexir/resources/results` folder. The name of this file will depend on the settings mentioned in the configuration file (`conf/exper_conf.py`). You will need this pickle file for the next step:

```commandline
python reasoning_with_vectors/experiments/analysis.py
```

Remember to specify the generated pickle file in the main section of `experiments/analysis.py`. For example:

```
if __name__ == '__main__':
    the_results_file = configuration.results_folder + 'dp4_Sum_Max_Min.pkl'
    obj = Percentile(the_results_file)
    obj.evaluate_performance_on_partitioned_dataset()
```

### Running the Condensed Approach

To evaluate the Condensed Approach, change `evaluation_model` in the configuration file (`conf/exper_conf.py`) to the following:

```commandline
evaluation_model = 'CompositionModel'
```

The Condensed Model checkpoints are available in folder: `/scratchtest/c.scmnk4/elexir/resources/learned_models`


However, if you want to train the Condensed Model again then do the following: 

* To create a dataset for training the Condensed Model, run:

```commandline
python reasoning_with_vectors/experiments/data_processor.py
```

* Train the Condensed Model on the created training dataset with:

```commandline
python reasoning_with_vectors/experiments/training.py
```

* Similar to Direct approach, to evaluate the Condensed approach, first run: 

```commandline
python reasoning_with_vectors/experiments/evaluation.py
```

This will create a pickle file in the `/scratchtest/c.scmnk4/elexir/resources/results` folder. You will need this pickle file for the next step:

```commandline
python reasoning_with_vectors/experiments/analysis.py
```

Remember to specify the generated pickle file in the main section of `experiments/analysis.py`.

## Training the importance classifier

To train the importance classifier do the following:

```commandline
python reasoning_with_vectors/importance/importance_filter_training.py
```


