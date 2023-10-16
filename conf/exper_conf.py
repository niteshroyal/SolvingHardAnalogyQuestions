# General configuration
# ---------------------
logging_folder = "/home/nitesh/elexir/reasoning_with_vectors/logs"
knowledge_graph_file = "/scratch/c.scmnk4/elexir/resources/conceptnet_kb.pl"

# Configuration for importance_filter.py
# --------------------------------------
# classifier_model_path = '/scratch/c.scmnk4/elexir/resources/learned_models/importance_classifier.pkl'
# classifier_model_path = '/scratch/c.scmnk4/elexir/resources/learned_models/conceptnet_importance_classifier.pkl'
classifier_model_path = '/scratch/c.scmnk4/elexir/resources/learned_models/gpt4_high_quality_importance_classifier.pkl'

# Configuration for cache.py
# --------------------------
rocksdb_path = '/scratch/c.scmnk4/elexir/resources/relbert_embeddings.db'
rocksdb_lock = '/scratch/c.scmnk4/elexir/resources/rocksdb.lock'
rocksdb_path_eval = '/scratch/c.scmnk4/elexir/resources/relbert_embeddings_for_evaluation.db'
rocksdb_eval_lock = '/scratch/c.scmnk4/elexir/resources/rocksdb_eval.lock'

# Configuration for preprocessing.py
# ----------------------------------
get_related_concepts_using_word_embeddings_topn = 5

# Configuration for data_processor.py
# ----------------------------------
relbert_batch_size = 250
num_of_training_concept_pairs = 50
importance_threshold = 0.75
inverse = False

# Configuration for training.py
# -----------------------------
vector_space_dimension = 1024
inner_layer_dimension = 256
number_of_epochs = 1000
training_batch_size = 10000
model_save_path = '/scratch/c.scmnk4/elexir/resources/learned_models/'
learning_rate = 0.0025

# Configuration for evaluation.py
# -------------------------------
analogy_datasets_path = "/scratch/c.scmnk4/elexir/resources/analogy_test_dataset/analogy_test_dataset"
analogy_datasets = ['sat', 'u2', 'u4', 'bats', 'google', 'scan', 'ekar']
# analogy_datasets = ['bats']
# Configuration for kb_constructor.py
# -----------------------------------
importance_filter_threshold = 0.75
glove_file = "/scratch/c.scmnk4/elexir/resources/glove_numberbatch.pl"
glove_filtered_file = "/scratch/c.scmnk4/elexir/resources/glove_numberbatch_filtered.pl"
relative_init_file = "/scratch/c.scmnk4/elexir/resources/relative_init_kb.pl"

# Configuration for relation_clustering.py
# ----------------------------------------
n_clusters = 50
rocksdb_clustering_path = '/scratch/c.scmnk4/elexir/resources/relbert_embeddings_for_clustering.db'
rocksdb_clustering_lock = '/scratch/c.scmnk4/elexir/resources/rocksdb_for_clustering.lock'
enriched_kb_with_clusters = f'/scratch/c.scmnk4/elexir/resources/enriched_kb_with_{n_clusters}clusters.pl'
n_components = 64
min_cluster_size = 3
clustering_batch_size = 500000
pca_path = '/scratch/c.scmnk4/elexir/resources/learned_models/incrementalPCA.pkl'
clustering_path = '/scratch/c.scmnk4/elexir/resources/learned_models/clusterer.pkl'
mini_batch_kmeans_batch_size = 25

# Configuration for enriched_kb.py
# --------------------------------
qa_vocab_file = "/scratch/c.scmnk4/elexir/resources/qa_concepts.txt"
original_conceptnet_used_in_qagnn = "/scratch/c.scmnk4/elexir/resources/conceptnet-assertions-5.6.0.csv"
enriched_conceptnet = f'/scratch/c.scmnk4/elexir/resources/enriched-conceptnet-5.6.0-' \
                      f'with-importance-thres-{importance_filter_threshold}.csv'
enriched_conceptnet_with_clusters = f'/scratch/c.scmnk4/elexir/resources/enriched-conceptnet-5.6.0-' \
                                    f'with-importance-thres-{importance_filter_threshold}-clusters-{n_clusters}.csv'

# Configuration for leveldb_cache.py
# ----------------------------------
leveldb_path = '/scratch/c.scmnk4/elexir/resources/leveldb_relbert_embeddings_for_clustering'
leveldb_lock = '/scratch/c.scmnk4/elexir/resources/leveldb.lock'

# Configuration for lmdb_cache.py
# ----------------------------------
lmdb_path = '/scratch/c.scmnk4/elexir/resources/lmdb_store'

# Configuration for set_transformer.py
# ------------------------------------
num_inds = 4
num_heads = 4

# Configuration for experiment1_data_processor.py
# -----------------------------------------------

# # Consider only ConceptNet edges for paths. (CN)
# dataset_preparation_approch_for_composition_model = 1

# (CN + Smoothing)
# dataset_preparation_approch_for_composition_model = 1.5

# # Compute 250 nearest neighbours from Glove and 250 from Numberbatch.
# # Then filter them using importance classifier. (WE)
# dataset_preparation_approch_for_composition_model = 2
#
# # WE + CN
# dataset_preparation_approch_for_composition_model = 3
#
# # WE + CN + Smoothing
dataset_preparation_approch_for_composition_model = 4
#
# # WE + filter(CN + Smoothing)
# dataset_preparation_approch_for_composition_model = 5

# # Same as approach 4, difference is that target pair is selected using importance classifier trained on ConceptNet
# dataset_preparation_approch_for_composition_model = 6
#
# # Same as approach 4, difference is that target pair is selected without using any importance classifier.
# dataset_preparation_approch_for_composition_model = 7


training_dataset = f"/scratch/c.scmnk4/elexir/resources/training/training_data_for_" \
                   f"composition_model_appr_{dataset_preparation_approch_for_composition_model}.pkl"
clear_redis_cache_in_the_start = True
training_concept_pairs_file = "/scratch/c.scmnk4/elexir/resources/training_concept_pairs.txt"

if dataset_preparation_approch_for_composition_model == 6:
    training_concept_pairs_file = "/scratch/c.scmnk4/elexir/resources/training_concept_pairs_dp6.txt"
    classifier_model_path = '/scratch/c.scmnk4/elexir/resources/learned_models/conceptnet_importance_classifier.pkl'
elif dataset_preparation_approch_for_composition_model == 7:
    training_concept_pairs_file = "/scratch/c.scmnk4/elexir/resources/training_concept_pairs_dp7.txt"

# Configuration for experiment1_training.py
# -----------------------------------------

# composition_model = 'ComplexCompositionModel'
# composition_model = 'CompositionModelInverse'
composition_model = 'CompositionModel'
# composition_model = 'SetTransformer'

num_runs = 1

# Configuration for experiment1_evaluation.py
# -------------------------------------------
# evaluation_model = 'Relation_Types'
evaluation_model = 'Sum_Max_Min'
# evaluation_model = 'Sum_Max_Sum'
# evaluation_model = 'Sum_Max_Hidden'
# evaluation_model = 'GPT-4'
# evaluation_model = 'GPT-3.5-Turbo'
# evaluation_model = 'CompositionModel'

llm_correct_predictions_file = ''
llm_incorrect_predictions_file = ''
if evaluation_model == 'GPT-4':
    llm_correct_predictions_file = '/scratch/c.scmnk4/elexir/resources/results/gpt-4_correct_predictions.txt'
    llm_incorrect_predictions_file = '/scratch/c.scmnk4/elexir/resources/results/gpt-4_incorrect_predictions.txt'
elif evaluation_model == 'GPT-3.5-Turbo':
    llm_correct_predictions_file = '/scratch/c.scmnk4/elexir/resources/results/' \
                                   'gpt-3.5-turbo_correct_predictions.txt'
    llm_incorrect_predictions_file = '/scratch/c.scmnk4/elexir/resources/results/' \
                                     'gpt-3.5-turbo_incorrect_predictions.txt'
else:
    pass

results_folder = '/scratch/c.scmnk4/elexir/resources/results/'
