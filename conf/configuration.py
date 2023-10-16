import os
import importlib.util

dir_path = os.path.dirname(os.path.realpath(__file__))
configuration_file_to_consider = os.path.join(dir_path, "exper_conf.py")


def load_module_from_file(filepath):
    spec = importlib.util.spec_from_file_location("conf", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module_from_file(configuration_file_to_consider)


logging_folder = config.logging_folder
knowledge_graph_file = config.knowledge_graph_file

classifier_model_path = config.classifier_model_path

rocksdb_path = config.rocksdb_path
rocksdb_lock = config.rocksdb_lock
rocksdb_path_eval = config.rocksdb_path_eval
rocksdb_eval_lock = config.rocksdb_eval_lock

get_related_concepts_using_word_embeddings_topn = config.get_related_concepts_using_word_embeddings_topn

relbert_batch_size = config.relbert_batch_size
num_of_training_concept_pairs = config.num_of_training_concept_pairs
importance_threshold = config.importance_threshold
inverse = config.inverse
training_dataset = config.training_dataset

vector_space_dimension = config.vector_space_dimension
inner_layer_dimension = config.inner_layer_dimension
number_of_epochs = config.number_of_epochs
training_batch_size = config.training_batch_size
model_save_path = config.model_save_path
learning_rate = config.learning_rate

analogy_datasets_path = config.analogy_datasets_path
analogy_datasets = config.analogy_datasets

importance_filter_threshold = config.importance_filter_threshold
glove_file = config.glove_file
glove_filtered_file = config.glove_filtered_file
relative_init_file = config.relative_init_file

n_clusters = config.n_clusters
rocksdb_clustering_path = config.rocksdb_clustering_path
rocksdb_clustering_lock = config.rocksdb_clustering_lock
enriched_kb_with_clusters = config.enriched_kb_with_clusters
n_components = config.n_components
min_cluster_size = config.min_cluster_size
clustering_batch_size = config.clustering_batch_size
pca_path = config.pca_path
clustering_path = config.clustering_path
mini_batch_kmeans_batch_size = config.mini_batch_kmeans_batch_size

qa_vocab_file = config.qa_vocab_file
original_conceptnet_used_in_qagnn = config.original_conceptnet_used_in_qagnn
enriched_conceptnet = config.enriched_conceptnet
enriched_conceptnet_with_clusters = config.enriched_conceptnet_with_clusters

lmdb_path = config.lmdb_path

num_inds = config.num_inds
num_heads = config.num_heads

dataset_preparation_approch_for_composition_model = config.dataset_preparation_approch_for_composition_model
clear_redis_cache_in_the_start = config.clear_redis_cache_in_the_start
training_concept_pairs_file = config.training_concept_pairs_file

composition_model = config.composition_model
num_runs = config.num_runs

evaluation_model = config.evaluation_model
results_folder = config.results_folder

llm_correct_predictions_file = config.llm_correct_predictions_file
llm_incorrect_predictions_file = config.llm_incorrect_predictions_file
