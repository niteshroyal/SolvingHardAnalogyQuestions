import re
import os
import ast
import pickle
import logging

import numpy as np
import unidecode
from relbert import cosine_similarity

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.importance.importance_filter import Classifier
from reasoning_with_vectors.experiments.data_processor import DataProcessor
from reasoning_with_vectors.experiments.training import PathLength2Training
from reasoning_with_vectors.core.utils import read_datasets


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


def get_lookup_key(dataset, stem, choices):
    query = '_'.join(stem)
    choices = '__'.join('_'.join(choice) for choice in choices)
    key = 'dataset_{}__query__{}__choices__{}'.format(dataset, query, choices)
    return key


class Common:
    def __init__(self):
        self.model = None
        self.classifier = None
        self.get_model()
        self.local_cache = dict()
        self.local_related_concepts = dict()
        self.relbert_embeddings_for_concept_pairs = dict()
        self.composition_model_embeddings_for_concept_pairs = dict()
        self.concepts_in_path_length_2 = dict()
        self.results = dict()
        self.composition = None

    def store_meta_data(self, key, value, dictionary):
        d = None
        if dictionary == 'relbert':
            d = self.relbert_embeddings_for_concept_pairs
        elif dictionary == 'composition':
            d = self.composition_model_embeddings_for_concept_pairs
        elif dictionary == 'related':
            d = self.concepts_in_path_length_2
        else:
            pass
        if key in d:
            pass
        else:
            d[key] = value

    def get_model(self):
        self.model = DataProcessor()
        self.model.set_path_finder_approach()
        self.classifier = Classifier()
        self.classifier.load_model()

    def similarity_approach(self, pair1, pair2):
        pass


class Evaluation(Common):
    def __init__(self, run_id=0):
        super().__init__()
        self.run_id = run_id
        self.result_file = os.path.join(configuration.results_folder,
                                        f"dp{configuration.dataset_preparation_approch_for_composition_model}_"
                                        f"{configuration.evaluation_model}.pkl")
        if configuration.evaluation_model in ['CompositionModel', 'Sum_Max_Hidden']:
            self.composition = PathLength2Training(self.run_id)
            self.composition.load_model()
            self.result_file = os.path.join(configuration.results_folder,
                                            f"evaluation_run{self.run_id}_"
                                            f"dp{configuration.dataset_preparation_approch_for_composition_model}_"
                                            f"dim{configuration.inner_layer_dimension}_"
                                            f"{configuration.evaluation_model}.pkl")

    def similarity_approach(self, stem_embd, choice):
        choice_embd = self.model.get_embedding_for_one_pair(choice)
        cosine = cosine_similarity(stem_embd, choice_embd)
        choice_imp = self.classifier.importance(choice_embd)
        return [cosine, choice_imp]

    def similarity_based_on_types(self, pair1, pair2):
        [a, b] = pair1
        [c, d] = pair2
        sim = 0
        qry = f'same_relation_types("{a}", "{b}", "{c}", "{d}")'
        for sol in self.model.prolog.query(qry):
            sim = 1
            break
        return sim

    def relation_types_approach(self, stem, choice):
        if configuration.dataset_preparation_approch_for_composition_model != 1:
            raise Exception('dataset_preparation_approch_for_composition_model should be 1')
        self.local_cache.clear()
        self.local_related_concepts.clear()
        [a, b] = stem
        related_ab = self.get_from_local_related_concepts(a, b)
        self.store_meta_data((a, b), related_ab, 'related')
        self.store_meta_data((a, b), self.get_from_local_cache([a, b]), 'relbert')
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.get_from_local_related_concepts(c, d)
            self.store_meta_data((c, d), related_cd, 'related')
            self.store_meta_data((c, d), self.get_from_local_cache([c, d]), 'relbert')
            logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', c, d, str(related_cd))
            if len(related_cd) > 0:
                triangle = True
        if len(related_ab) > 0 and triangle:
            predictions = []
            for [c, d] in choice:
                related_cd = self.get_from_local_related_concepts(c, d)
                tempsim = []
                for item1 in related_ab:
                    temp = [0]
                    for item2 in related_cd:
                        temp.append(min([self.similarity_based_on_types([a, item1], [c, item2]),
                                         self.similarity_based_on_types([item1, b], [item2, d])]))
                    tempsim.append(max(temp))
                tempsim = np.mean(tempsim)
                predictions.append(tempsim)
            prediction = predictions.index(max(predictions))
        else:
            sim = [cosine_similarity(self.get_from_local_cache([a, b]),
                                     self.get_from_local_cache([c, d])) for [c, d] in choice]
            prediction = sim.index(max(sim))
        return prediction

    def sum_max_min_approach(self, stem, choice):
        self.local_cache.clear()
        self.local_related_concepts.clear()
        [a, b] = stem
        related_ab = self.get_from_local_related_concepts(a, b)
        self.store_meta_data((a, b), related_ab, 'related')
        self.store_meta_data((a, b), self.get_from_local_cache([a, b]), 'relbert')
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.get_from_local_related_concepts(c, d)
            self.store_meta_data((c, d), related_cd, 'related')
            self.store_meta_data((c, d), self.get_from_local_cache([c, d]), 'relbert')
            logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', c, d, str(related_cd))
            if len(related_cd) > 0:
                triangle = True
        if len(related_ab) > 0 and triangle:
            predictions = []
            for [c, d] in choice:
                related_cd = self.get_from_local_related_concepts(c, d)
                tempsim = []
                for item1 in related_ab:
                    temp = [-2]
                    for item2 in related_cd:
                        if configuration.evaluation_model == 'Sum_Max_Min':
                            temp.append(min([cosine_similarity(self.get_from_local_cache([a, item1]),
                                                               self.get_from_local_cache([c, item2])),
                                             cosine_similarity(self.get_from_local_cache([item1, b]),
                                                               self.get_from_local_cache([item2, d]))]))
                        elif configuration.evaluation_model == 'Sum_Max_Sum':
                            temp.append(cosine_similarity([x + y for x, y in
                                                           zip(self.get_from_local_cache([a, item1]),
                                                               self.get_from_local_cache([item1, b]))],
                                                          [x + y for x, y in
                                                           zip(self.get_from_local_cache([c, item2]),
                                                               self.get_from_local_cache([item2, d]))]))
                        else:
                            raise Exception('Not a valid evaluation_model in the configuration')
                    tempsim.append(max(temp))
                tempsim = np.mean(tempsim)
                predictions.append(tempsim)
            prediction = predictions.index(max(predictions))
        else:
            sim = [cosine_similarity(self.get_from_local_cache([a, b]),
                                     self.get_from_local_cache([c, d])) for [c, d] in choice]
            prediction = sim.index(max(sim))
        return prediction

    def composition_approach(self, stem, choice):
        self.local_cache.clear()
        self.local_related_concepts.clear()
        [a, b] = stem
        related_ab = self.get_from_local_related_concepts(a, b)
        self.store_meta_data((a, b), related_ab, 'related')
        self.store_meta_data((a, b), self.get_from_local_cache([a, b]), 'relbert')
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.get_from_local_related_concepts(c, d)
            self.store_meta_data((c, d), related_cd, 'related')
            self.store_meta_data((c, d), self.get_from_local_cache([c, d]), 'relbert')
            logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', c, d, str(related_cd))
            if len(related_cd) > 0:
                triangle = True
        if len(related_ab) > 0 and triangle:
            predictions = []
            xz_list = []
            zy_list = []
            segment_ids = []
            for item in related_ab:
                xz_list.append(self.get_from_local_cache([a, item]))
                zy_list.append(self.get_from_local_cache([item, b]))
                segment_ids.append(0)
            ab_predicted_embd = self.composition.run_prediction(xz_list, zy_list, segment_ids)[0]
            self.store_meta_data((a, b), ab_predicted_embd, 'composition')
            for [c, d] in choice:
                related_cd = self.get_from_local_related_concepts(c, d)
                xz_list = []
                zy_list = []
                segment_ids = []
                for item in related_cd:
                    xz_list.append(self.get_from_local_cache([c, item]))
                    zy_list.append(self.get_from_local_cache([item, d]))
                    segment_ids.append(0)
                if len(related_cd) > 0:
                    cd_predicted_embd = self.composition.run_prediction(xz_list, zy_list, segment_ids)[0]
                    self.store_meta_data((c, d), cd_predicted_embd, 'composition')
                    tempsim = cosine_similarity(ab_predicted_embd, cd_predicted_embd)
                else:
                    tempsim = -1
                predictions.append(tempsim)
            prediction = predictions.index(max(predictions))
        else:
            sim = [cosine_similarity(self.get_from_local_cache([a, b]),
                                     self.get_from_local_cache([c, d])) for [c, d] in choice]
            prediction = sim.index(max(sim))
        return prediction

    def composition_mean_max_approach(self, stem, choice):
        self.local_cache.clear()
        self.local_related_concepts.clear()
        [a, b] = stem
        related_ab = self.get_from_local_related_concepts(a, b)
        self.store_meta_data((a, b), related_ab, 'related')
        self.store_meta_data((a, b), self.get_from_local_cache([a, b]), 'relbert')
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.get_from_local_related_concepts(c, d)
            self.store_meta_data((c, d), related_cd, 'related')
            self.store_meta_data((c, d), self.get_from_local_cache([c, d]), 'relbert')
            logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', c, d, str(related_cd))
            if len(related_cd) > 0:
                triangle = True
        if len(related_ab) > 0 and triangle:
            predictions = []
            xz_list1 = []
            zy_list1 = []
            segment_ids = []
            for item in related_ab:
                xz_list1.append(self.get_from_local_cache([a, item]))
                zy_list1.append(self.get_from_local_cache([item, b]))
                segment_ids.append(0)
            for [c, d] in choice:
                related_cd = self.get_from_local_related_concepts(c, d)
                xz_list2 = []
                zy_list2 = []
                segment_ids = []
                for item in related_cd:
                    xz_list2.append(self.get_from_local_cache([c, item]))
                    zy_list2.append(self.get_from_local_cache([item, d]))
                    segment_ids.append(0)
                if len(related_cd) > 0:
                    tempsim = self.composition.run_path_based_similarity(xz_list1, zy_list1, xz_list2, zy_list2)
                else:
                    tempsim = -1
                predictions.append(tempsim)
            prediction = predictions.index(max(predictions))
        else:
            sim = [cosine_similarity(self.get_from_local_cache([a, b]),
                                     self.get_from_local_cache([c, d])) for [c, d] in choice]
            prediction = sim.index(max(sim))
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
            if configuration.dataset_preparation_approch_for_composition_model in [1, 1.5, 2, 3, 4, 6, 7]:
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

    def predictions_per_dataset(self, data, dataset_name):
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
            appr_predicted_answer = None
            if configuration.evaluation_model in ['Sum_Max_Min', 'Sum_Max_Sum']:
                appr_predicted_answer = self.sum_max_min_approach(stem, choice)
            elif configuration.evaluation_model == 'CompositionModel':
                appr_predicted_answer = self.composition_approach(stem, choice)
            elif configuration.evaluation_model == 'Relation_Types':
                appr_predicted_answer = self.relation_types_approach(stem, choice)
            elif configuration.evaluation_model == 'Sum_Max_Hidden':
                appr_predicted_answer = self.composition_mean_max_approach(stem, choice)
            else:
                pass
            key = get_lookup_key(dataset_name, stem, choice)
            self.results[key] = int(appr_predicted_answer)
            counter = counter + 1
            if counter % 100 == 1:
                logging.info(f'Evaluation Progress: Dataset = {dataset_name}, '
                             f'Method = {configuration.evaluation_model}, {counter} out of {data_len} processed')

    def predictions(self):
        for dataset_name in configuration.analogy_datasets:
            data = read_datasets(dataset_name, approach='test_and_valid')
            self.predictions_per_dataset(data, dataset_name)

    def record_results(self):
        results = [{'relbert': self.relbert_embeddings_for_concept_pairs},
                   {'composition': self.composition_model_embeddings_for_concept_pairs},
                   {'related': self.concepts_in_path_length_2},
                   {'predictions': self.results}]
        with open(self.result_file, 'wb') as f:
            pickle.dump(results, f)


class LLMEvaluation:
    def __init__(self):
        self.model = DataProcessor()
        self.result_file = os.path.join(configuration.results_folder, f"{configuration.evaluation_model}.pkl")
        self.results = dict()
        self.relbert_embeddings_for_concept_pairs = dict()

    def store_relbert_embeddings(self, key):
        if key in self.relbert_embeddings_for_concept_pairs:
            pass
        else:
            (a, b) = key
            self.relbert_embeddings_for_concept_pairs[key] = self.model.get_embedding_for_one_pair([a, b])

    def store_questions_relbert_embeddings(self, query, choices):
        [a, b] = query
        a = unidecode.unidecode(a.lower())
        b = unidecode.unidecode(b.lower())
        self.store_relbert_embeddings((a, b))
        for [c, d] in choices:
            c = unidecode.unidecode(c.lower())
            d = unidecode.unidecode(d.lower())
            self.store_relbert_embeddings((c, d))

    def record_llm_results(self):
        llm_prediction_files = [configuration.llm_correct_predictions_file,
                                configuration.llm_incorrect_predictions_file]
        for filename in llm_prediction_files:
            with open(filename, 'r') as file:
                content = file.read().split('\n')
                for entry in content:
                    if 'Dataset = ' in entry[0:10]:
                        dataset_name = re.search(r'Dataset = (.*?),', entry).group(1)
                        query_raw = re.search(r'Query = (\[.*?\]),', entry).group(1)
                        query = ast.literal_eval(query_raw)
                        choices_raw = re.search(r'Choices = (\[.*?\]), Actual answer', entry).group(1)
                        choices = ast.literal_eval(choices_raw)
                        predicted_answer = re.search(r'Predicted answer = (.*?),', entry).group(1)
                        if predicted_answer == 'None':
                            predicted_answer = '-1'
                        key = get_lookup_key(dataset_name, query, choices)
                        self.results[key] = int(predicted_answer)
                        self.store_questions_relbert_embeddings(query, choices)
        results = [{'relbert': self.relbert_embeddings_for_concept_pairs},
                   {'composition': dict()},
                   {'related': dict()},
                   {'predictions': self.results}]
        with open(self.result_file, 'wb') as f:
            pickle.dump(results, f)


def evaluate_models_for_experiment1():
    if configuration.evaluation_model == 'GPT-4' or configuration.evaluation_model == 'GPT-3.5-Turbo':
        obj = LLMEvaluation()
        obj.record_llm_results()
    else:
        num_runs = configuration.num_runs
        for i in range(0, num_runs):
            obj = Evaluation(i)
            obj.predictions()
            obj.record_results()


if __name__ == '__main__':
    initialization()
    evaluate_models_for_experiment1()
