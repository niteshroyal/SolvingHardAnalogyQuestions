import os
import random
import logging
# import hdbscan
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans

from pyswip import Prolog

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from pythonProject.research.reasoning_with_vectors.conf import configuration
from cache import Cache


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


header = f'''%%% -*- Mode: Prolog; -*-
% Concept pairs and cluster assignments are represented using the relation t/3.   

'''


def get_all_enriched_conceptnet_links():
    prolog = Prolog()
    prolog.consult(configuration.knowledge_graph_file)
    conceptnet_links = set()
    qry = 'enriched_edge(X,Y).'
    for sol in prolog.query(qry):
        c1 = sol['X'].decode('UTF-8')
        c2 = sol['Y'].decode('UTF-8')
        conceptnet_links.add((c1, c2))
    return sorted(conceptnet_links)


class RelationClustering(Cache):
    def __init__(self):
        super().__init__(read_only=False, db_path=configuration.rocksdb_clustering_path,
                         lock_path=configuration.rocksdb_clustering_lock)
        self.all_keys = None
        self.clusters = None
        self.ipca = IncrementalPCA(n_components=configuration.n_components)
        # self.clusterer = hdbscan.HDBSCAN(min_cluster_size=configuration.min_cluster_size)
        self.clusterer = MiniBatchKMeans(n_clusters=configuration.n_clusters,
                                         batch_size=configuration.mini_batch_kmeans_batch_size)

    def cache_all_embeddings(self):
        conceptnet_links = get_all_enriched_conceptnet_links()
        conceptnet_links_list = []
        for (c1, c2) in conceptnet_links:
            conceptnet_links_list.append([c1, c2])
        self.cache_embedding_with_batch_size(conceptnet_links_list, configuration.relbert_batch_size)

    def get_all_keys(self):
        self.all_keys = []
        all_keys = self.all_stored_relbert_embds()
        for [c1, c2] in all_keys:
            self.all_keys.append((c1, c2))
        logging.info(f'Total number of keys = {len(self.all_keys)}')

    def incremental_pca(self, vectors):
        self.ipca.partial_fit(np.array(vectors))

    def fit_pca(self):
        count = 0
        working_data = []
        for (c1, c2) in self.all_keys:
            working_data.append(self.get_embedding_for_one_pair([c1, c2]))
            count += 1
            if count >= configuration.clustering_batch_size:
                logging.info('Fitting PCA on batch of size %s', count)
                self.incremental_pca(normalize(np.vstack(working_data)))
                count = 0
                working_data = []
        if count > 0:
            logging.info('Fitting PCA on final batch of size %s', count)
            self.incremental_pca(normalize(np.vstack(working_data)))
        self.save_ipca_model()

    def transformed_data(self):
        transformed_data = []
        count = 0
        working_data = []
        for (c1, c2) in self.all_keys:
            working_data.append(self.get_embedding_for_one_pair([c1, c2]))
            count += 1
            if count >= configuration.clustering_batch_size:
                transformed_data.append(self.ipca.transform(normalize(np.vstack(working_data))))
                count = 0
                working_data = []
        if count > 0:
            transformed_data.append(self.ipca.transform(normalize(np.vstack(working_data))))
        return np.concatenate(transformed_data, axis=0)

    def clustering(self):
        logging.info('Starting the clustering procedure')
        data = np.array(self.transformed_data())
        self.clusterer.fit(data)
        labels = self.clusterer.labels_
        self.clusters = dict(zip(self.all_keys, labels))
        logging.info('Finished the clustering procedure')
        self.save_clustering_model()

    def train_models(self):
        # self.fit_pca()
        self.load_ipca_model()
        self.clustering()

    def enriched_kb_with_clusters(self):
        self.get_all_keys()
        self.train_models()
        handle = open(configuration.enriched_kb_with_clusters, 'w')
        handle.write(header)
        for key in self.clusters:
            (c1, c2) = key
            cluster = self.clusters[key]
            handle.write('t("' + str(c1) + '", "' + str(c2) + '", "c' + str(cluster) + '").\n')
        handle.close()

    def save_clustering_model(self):
        with open(configuration.clustering_path, 'wb') as f:
            pickle.dump(self.clusterer, f)
        logging.info('Clustering model have been saved successfully')

    def load_clustering_model(self):
        with open(configuration.clustering_path, 'rb') as f:
            self.clusterer = pickle.load(f)
        logging.info('Clustering model have been loaded successfully')

    def load_ipca_model(self):
        with open(configuration.pca_path, 'rb') as f:
            self.ipca = pickle.load(f)
        logging.info('PCA models have been loaded successfully')

    def save_ipca_model(self):
        with open(configuration.pca_path, 'wb') as f:
            pickle.dump(self.ipca, f)
        logging.info('PCA models have been saved successfully')

    def plot_k_distance_graph(self, minPts=5, sample_size=500000):
        self.get_all_keys()
        self.fit_pca()
        data = self.transformed_data()
        num_samples = min(sample_size, len(data))
        sample_indices = random.sample(range(len(data)), num_samples)
        data = data[sample_indices]
        nn = NearestNeighbors(n_neighbors=minPts)
        nn.fit(data)
        distances, indices = nn.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        plt.plot(distances)
        plt.title("k-distance Graph")
        plt.ylabel("k-distances")
        plt.grid(True)
        plt.savefig('k_distance_graph.png')
        plt.close()


if __name__ == '__main__':
    initialization()
    obj = RelationClustering()
    # obj.cache_all_embeddings()
    obj.enriched_kb_with_clusters()
    # obj.plot_k_distance_graph()
