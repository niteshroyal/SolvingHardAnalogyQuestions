import os
import random
import logging
import pickle
import time

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.core.preprocessing import Processor
from reasoning_with_vectors.importance.importance_filter import Classifier


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def write_training_data(data, counter):
    filename = os.path.join(os.path.dirname(configuration.training_dataset),
                            os.path.splitext(os.path.basename(configuration.training_dataset))[0] +
                            '_part_' + str(counter) + '.pkl')
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TrainingDataProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.training_concept_pairs = None
        self.importance_filter = None

    def get_filtered_concept_pairs(self, concept_pairs):
        training_concept_pairs = []
        start_time = time.time()
        embeddings = self.get_embedding(concept_pairs)
        logging.info(f'Time taken to get {len(concept_pairs)} embeddings is {time.time() - start_time} secs')
        for i in range(0, len(embeddings)):
            importance = self.importance_filter.importance(embeddings[i])
            [c1, c2] = concept_pairs[i]
            if importance > configuration.importance_threshold:
                training_concept_pairs.append([c1, c2])
        return training_concept_pairs

    def get_training_concept_pairs(self):
        self.importance_filter = Classifier()
        self.importance_filter.load_model()
        links = self.get_all_conceptnet_links()
        links = list(links)
        random.seed(42)
        random.shuffle(links)
        training_concept_pairs = []
        counter = 0
        temp_links = []
        logging.info('Number of ConceptNet links to filter = %d', len(links))
        for (c1, c2) in links:
            temp_links.append([c1, c2])
            if len(temp_links) >= configuration.relbert_batch_size:
                start_time = time.time()
                training_concept_pairs += self.get_filtered_concept_pairs(temp_links)
                logging.info(f'Time taken to get {len(temp_links)} filtered embeddings is '
                             f'{time.time() - start_time} secs')
                temp_links = []
            counter += 1
            if counter % 1000 == 0:
                logging.info('Number of links filtered = %d', counter)
            if len(training_concept_pairs) >= configuration.num_of_training_concept_pairs:
                break
        if len(temp_links) > 0:
            training_concept_pairs += self.get_filtered_concept_pairs(temp_links)
        self.training_concept_pairs = training_concept_pairs
        return training_concept_pairs

    def get_training_concept_pairs_from_db(self):
        logging.info(f'Could not load target concept pairs from {configuration.training_concept_pairs_file}')
        logging.info(f'Now going to extract target concept pairs from {configuration.lmdb_path}')
        self.importance_filter = Classifier()
        self.importance_filter.load_model()
        self.training_concept_pairs = []
        counter = 0
        for concept_pair, embedding in self.iter_stored_relbert_embds():
            counter += 1
            [c1, c2] = concept_pair
            if self.is_conceptnet_link(c1, c2):
                if configuration.dataset_preparation_approch_for_composition_model == 7:
                    self.training_concept_pairs.append(concept_pair)
                else:
                    importance = self.importance_filter.importance(embedding)
                    if importance > configuration.importance_threshold:
                        self.training_concept_pairs.append(concept_pair)
            if counter % 10000 == 0:
                logging.info(f'Total number of links filtered = {counter}')
            if len(self.training_concept_pairs) >= configuration.num_of_training_concept_pairs:
                break
        logging.info(f'Total number of links filtered = {counter}')
        logging.info(f'Number of extracted training concept pairs = {len(self.training_concept_pairs)}')
        return self.training_concept_pairs

    def extract_training_data(self):
        required_relbert_vectors = set()
        for [c1, c2] in self.training_concept_pairs:
            rel_concepts = self.get_related_concepts(c1, c2)
            for item in rel_concepts:
                required_relbert_vectors.add((c1, item))
                required_relbert_vectors.add((item, c2))
                if configuration.inverse:
                    required_relbert_vectors.add((item, c1))
                    required_relbert_vectors.add((c2, item))
        required_relbert_vectors_list = []
        for (c1, c2) in required_relbert_vectors:
            required_relbert_vectors_list.append([c1, c2])
        self.cache_embedding_with_batch_size(required_relbert_vectors_list, configuration.relbert_batch_size)
        counter = 0
        training_data = []
        for [c1, c2] in self.training_concept_pairs:
            rel_concepts = self.get_related_concepts(c1, c2)
            if not rel_concepts:
                continue
            xy = self.get_embedding_for_one_pair([c1, c2])
            xzy = []
            for item in rel_concepts:
                if configuration.inverse:
                    xzy.append([item, self.get_embedding_for_one_pair([c1, item]),
                                self.get_embedding_for_one_pair([item, c2]),
                                self.get_embedding_for_one_pair([item, c1]),
                                self.get_embedding_for_one_pair([c2, item])])
                else:
                    xzy.append([item, self.get_embedding_for_one_pair([c1, item]),
                                self.get_embedding_for_one_pair([item, c2])])
            training_data.append([c1, c2, xy, xzy])
            if len(training_data) >= 1000:
                counter += 1
                write_training_data(training_data, counter)
                training_data = []
        if len(training_data) > 0:
            counter += 1
            write_training_data(training_data, counter)

    def is_conceptnet_link(self, c1, c2):
        answer = False
        qry = f'considered_link("{c1}", "{c2}").'
        for sol in self.prolog.query(qry):
            answer = True
            break
        return answer

    def get_all_conceptnet_links(self):
        conceptnet_links = set()
        qry = 'considered_link(X,Y).'
        for sol in self.prolog.query(qry):
            c1 = sol['X'].decode('UTF-8')
            c2 = sol['Y'].decode('UTF-8')
            conceptnet_links.add((c1, c2))
            # if len(conceptnet_links) > 10:
            #     break
        return sorted(conceptnet_links)

    def get_all_conceptnet_links_plus_reverse(self):
        conceptnet_links = set()
        qry = 'considered_link(X,Y).'
        for sol in self.prolog.query(qry):
            c1 = sol['X'].decode('UTF-8')
            c2 = sol['Y'].decode('UTF-8')
            conceptnet_links.add((c1, c2))
            conceptnet_links.add((c2, c1))
        return sorted(conceptnet_links)

    def store_all_conceptnet_relbert_vectors(self):
        conceptnet_links = self.get_all_conceptnet_links()
        conceptnet_links_list = []
        for (c1, c2) in conceptnet_links:
            conceptnet_links_list.append([c1, c2])
        self.cache_embedding_with_batch_size(conceptnet_links_list, configuration.relbert_batch_size)


if __name__ == '__main__':
    initialization()
    obj = TrainingDataProcessor()
    # obj.store_all_conceptnet_relbert_vectors()
    obj.get_training_concept_pairs_from_db()
    # obj.get_training_concept_pairs()
    obj.extract_training_data()
