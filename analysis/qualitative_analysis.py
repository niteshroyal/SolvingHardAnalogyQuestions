import logging
import os

import numpy as np
import unidecode
from relbert import cosine_similarity

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.core.utils import read_datasets
from reasoning_with_vectors.experiments.data_processor import DataProcessor
from reasoning_with_vectors.importance.importance_filter import Classifier


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            f"{os.path.splitext(os.path.basename(__file__))[0]}_"
                            f"dp{configuration.dataset_preparation_approch_for_composition_model}_"
                            f"{configuration.evaluation_model}.log")
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=log_file, filemode='w', level=logging.INFO)
    with open(configuration.configuration_file_to_consider, 'r') as handle:
        conf = handle.read()
    logging.info(conf)


class QualitativeAnalysis:
    def __init__(self):
        self.model = None
        self.classifier = None
        self.get_model()
        self.local_cache = dict()
        self.local_related_concepts = dict()

    def get_model(self):
        self.model = DataProcessor()
        self.model.set_path_finder_approach()
        self.classifier = Classifier()
        self.classifier.load_model()

    def sum_max_min_analysis(self, stem, choice, ans):
        self.local_cache.clear()
        self.local_related_concepts.clear()
        [a, b] = stem
        related_ab = self.get_from_local_related_concepts(a, b)
        logging.info(f'Query = {stem}, Choices = {choice}, Answer = {ans}')
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.get_from_local_related_concepts(c, d)
            logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', c, d, str(related_cd))
            if len(related_cd) > 0:
                triangle = True
        if len(related_ab) > 0 and triangle:
            predictions = []
            for [c, d] in choice:
                related_cd = self.get_from_local_related_concepts(c, d)
                tempsim = []
                query_interim = []
                for item1 in related_ab:
                    temp = [-2]
                    choice_interim = [(c, 'empty_choice_interim', d)]
                    for item2 in related_cd:
                        score = min([cosine_similarity(self.get_from_local_cache([a, item1]),
                                                       self.get_from_local_cache([c, item2])),
                                     cosine_similarity(self.get_from_local_cache([item1, b]),
                                                       self.get_from_local_cache([item2, d]))])
                        temp.append(score)
                        choice_interim.append((c, item2, d, score))
                    maximum = max(temp)
                    tempsim.append(maximum)
                    most_relevant_choice_interim = choice_interim[temp.index(maximum)]
                    query_interim.append(((a, item1, b), most_relevant_choice_interim))
                sorted_tempsim = np.argsort(tempsim).tolist()
                sorted_tempsim.reverse()
                for idx in sorted_tempsim:
                    logging.info(f'Matching query and choice paths = {query_interim[idx]}, ')
                tempsim = np.mean(tempsim)
                predictions.append(tempsim)
            prediction = predictions.index(max(predictions))
            logging.info(f'Predicted answer using direct approach = {prediction}')
        else:
            sim = [cosine_similarity(self.get_from_local_cache([a, b]),
                                     self.get_from_local_cache([c, d])) for [c, d] in choice]
            prediction = sim.index(max(sim))
            logging.info(f'Predicted answer using RelBERT approach = {prediction}')
        return prediction

    def get_from_local_cache(self, pair):
        [a, b] = pair
        key = (a, b)
        if key in self.local_cache:
            return self.local_cache[key]
        else:
            embedding = self.model.get_embedding_for_one_pair(pair)
            self.local_cache[key] = embedding
            return embedding

    def get_from_local_related_concepts(self, a, b):
        key = (a, b)
        if key in self.local_related_concepts:
            return self.local_related_concepts[key]
        else:
            related_concepts_temp = self.model.get_related_concepts(a, b)
            related_concepts = []
            if configuration.dataset_preparation_approch_for_composition_model in [1, 1.5, 2, 3, 4]:
                related_concepts = related_concepts_temp
            elif configuration.dataset_preparation_approch_for_composition_model in [5]:
                for item in related_concepts_temp:
                    if self.classifier.importance(self.get_from_local_cache([a, item])) > \
                            configuration.importance_threshold and \
                            self.classifier.importance(self.get_from_local_cache([item, b])) > \
                            configuration.importance_threshold:
                        related_concepts.append(item)
            else:
                logging.info('Valid data prepration approach not selected.')
        self.local_related_concepts[key] = related_concepts
        return related_concepts

    def analysis_per_dataset(self, data, dataset_name):
        data_len = len(data)
        counter = 0
        for item in data:
            stem = item['stem']
            [stem1, stem2] = stem
            stem1 = unidecode.unidecode(stem1.lower())
            stem2 = unidecode.unidecode(stem2.lower())
            stem = [stem1, stem2]
            choice = item['choice']
            temp = []
            for ch in choice:
                [ch1, ch2] = ch
                ch1 = unidecode.unidecode(ch1.lower())
                ch2 = unidecode.unidecode(ch2.lower())
                ch = [ch1, ch2]
                temp.append(ch)
            choice = temp
            self.sum_max_min_analysis(stem, choice, item['answer'])
            counter = counter + 1
            if counter % 100 == 1:
                logging.info(f'Evaluation Progress: Dataset = {dataset_name}, '
                             f'Method = {configuration.evaluation_model}, {counter} out of {data_len} processed')

    def anlysis(self):
        for dataset_name in configuration.analogy_datasets:
            data = read_datasets(dataset_name, approach='test_and_valid')
            self.analysis_per_dataset(data, dataset_name)


def qualitative_analysis():
    obj = QualitativeAnalysis()
    obj.anlysis()


if __name__ == '__main__':
    initialization()
    qualitative_analysis()
