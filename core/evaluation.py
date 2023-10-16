import logging
import os
import pickle

import numpy as np
import unidecode
from relbert import cosine_similarity

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.importance.importance_filter import Classifier
from reasoning_with_vectors.core.preprocessing import Processor
from reasoning_with_vectors.core.training import PathLength2Training
from reasoning_with_vectors.core.utils import read_datasets


def initialization():
    log_file = os.path.join(configuration.logging_folder, f"{os.path.splitext(os.path.basename(__file__))[0]}_"
                                                          f"sum_max_min.log")
    # log_file = os.path.join(configuration.logging_folder, f"{os.path.splitext(os.path.basename(__file__))[0]}_"
    #                                                       f"composition_{configuration.inner_layer_dimension}.log")
    # log_file = os.path.join(configuration.logging_folder, f"{os.path.splitext(os.path.basename(__file__))[0]}_"
    #                                                       f"composition_{configuration.inner_layer_dimension}_"
    #                                                       f"mean_max.log")
    # log_file = os.path.join(configuration.logging_folder, f"{os.path.splitext(os.path.basename(__file__))[0]}_"
    #                                                       f"transformer_{configuration.inner_layer_dimension}.log")
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=log_file, filemode='w', level=logging.INFO)
    with open(configuration.__file__, 'r') as handle:
        conf = handle.read()
    logging.info(conf)


class NoModel:
    def __init__(self):
        self.model = None
        self.classifier = None
        self.composition = None
        self.get_model()
        self.local_cache = dict()
        self.relbert_embeddings_for_concept_pairs = dict()
        self.composition_model_embeddings_for_concept_pairs = dict()
        self.concepts_in_path_length_2 = dict()

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
        # self.model = Processor(read_only=True, db_path=configuration.rocksdb_path_eval,
        #                        lock_path=configuration.rocksdb_eval_lock)
        self.model = Processor()
        self.classifier = Classifier()
        self.classifier.load_model()
        self.composition = PathLength2Training()
        self.composition.load_model()

    def similarity_approach(self, pair1, pair2):
        pass


class SimImportanceUnderstanding(NoModel):
    def similarity_approach(self, stem_embd, choice):
        choice_embd = self.model.get_embedding_for_one_pair(choice)
        cosine = cosine_similarity(stem_embd, choice_embd)
        choice_imp = self.classifier.importance(choice_embd)
        return [cosine, choice_imp]

    def second_approach_sum_max_min(self, stem, choice):
        self.local_cache.clear()
        [a, b] = stem
        related_ab = self.model.get_related_concepts(a, b)
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.model.get_related_concepts(c, d)
            logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', c, d, str(related_cd))
            if len(related_cd) > 0:
                triangle = True
        if len(related_ab) > 0 and triangle:
            predictions = []
            for [c, d] in choice:
                related_cd = self.model.get_related_concepts(c, d)
                tempsim = []
                for item1 in related_ab:
                    temp = [-2]
                    for item2 in related_cd:
                        temp.append(min([cosine_similarity(self.get_from_local_cache([a, item1]),
                                                           self.get_from_local_cache([c, item2])),
                                         cosine_similarity(self.get_from_local_cache([item1, b]),
                                                           self.get_from_local_cache([item2, d]))]))
                    tempsim.append(max(temp))
                tempsim = np.mean(tempsim)
                predictions.append(tempsim)
            prediction = predictions.index(max(predictions))
        else:
            sim = [cosine_similarity(self.get_from_local_cache([a, b]),
                                     self.get_from_local_cache([c, d])) for [c, d] in choice]
            prediction = sim.index(max(sim))
        return prediction

    def second_approach_composition(self, stem, choice):
        self.local_cache.clear()
        [a, b] = stem
        related_ab = self.model.get_related_concepts(a, b)
        self.store_meta_data((a, b), related_ab, 'related')
        self.store_meta_data((a, b), self.get_from_local_cache([a, b]), 'relbert')
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.model.get_related_concepts(c, d)
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
                related_cd = self.model.get_related_concepts(c, d)
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

    def second_approach_composition_mean_max(self, stem, choice):
        self.local_cache.clear()
        [a, b] = stem
        related_ab = self.model.get_related_concepts(a, b)
        self.store_meta_data((a, b), related_ab, 'related')
        self.store_meta_data((a, b), self.get_from_local_cache([a, b]), 'relbert')
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.model.get_related_concepts(c, d)
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
                related_cd = self.model.get_related_concepts(c, d)
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

    def second_approach_composition_with_inverse(self, stem, choice):
        self.local_cache.clear()
        [a, b] = stem
        related_ab = self.model.get_related_concepts(a, b)
        logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', a, b, str(related_ab))
        triangle = False
        for [c, d] in choice:
            related_cd = self.model.get_related_concepts(c, d)
            logging.info('Concept1: %s, Concept2: %s, Related_Concepts: %s', c, d, str(related_cd))
            if len(related_cd) > 0:
                triangle = True
        if len(related_ab) > 0 and triangle:
            predictions = []
            xz_list1 = []
            zy_list1 = []
            xz_list2 = []
            zy_list2 = []
            segment_ids = []
            for item in related_ab:
                xz_list1.append(self.get_from_local_cache([a, item]))
                zy_list1.append(self.get_from_local_cache([item, b]))
                xz_list2.append(self.get_from_local_cache([item, a]))
                zy_list2.append(self.get_from_local_cache([b, item]))
                segment_ids.append(0)
            ab_predicted_embd = self.composition.run_prediction(xz_list1, zy_list1, segment_ids, xz_list2, zy_list2)[0]
            for [c, d] in choice:
                related_cd = self.model.get_related_concepts(c, d)
                xz_list1 = []
                zy_list1 = []
                xz_list2 = []
                zy_list2 = []
                segment_ids = []
                for item in related_cd:
                    xz_list1.append(self.get_from_local_cache([c, item]))
                    zy_list1.append(self.get_from_local_cache([item, d]))
                    xz_list2.append(self.get_from_local_cache([item, c]))
                    zy_list2.append(self.get_from_local_cache([d, item]))
                    segment_ids.append(0)
                if len(related_cd) > 0:
                    cd_predicted_embd = self.composition.run_prediction(xz_list1, zy_list1, segment_ids,
                                                                        xz_list2, zy_list2)[0]
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

    def get_from_local_cache(self, pair):
        [a, b] = pair
        key = (a, b)
        if key in self.local_cache:
            return self.local_cache[key]
        else:
            embedding = self.model.get_embedding_for_one_pair(pair)
            self.local_cache[key] = embedding
            return embedding

    def clear_local_cache(self):
        self.local_cache.clear()


def analogy_dataset_for_sim_importance_understanding_with_second_appr(data, dataset_name, approach):
    method = approach.__class__.__name__
    data_len = len(data)
    counter = 0
    records = []
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
        # approach2_predicted_answer = approach.second_approach_composition_mean_max(stem, choice)
        # approach2_predicted_answer = approach.second_approach_composition(stem, choice)
        # approach2_predicted_answer = approach.second_approach_composition_with_inverse(stem, choice)
        approach2_predicted_answer = approach.second_approach_sum_max_min(stem, choice)
        answer = item['answer']
        stem_embd = approach.model.get_embedding_for_one_pair(stem)
        record = [approach.classifier.importance(stem_embd)]
        for c in choice:
            record = record + approach.similarity_approach(stem_embd, c)
        record = record + [approach2_predicted_answer]
        record = record + [answer]
        record = record + [item]
        records.append(record)
        counter = counter + 1
        if counter % 100 == 1:
            logging.info('Evaluation Progress: Dataset = %s, Method = %s, %s out of %s processed',
                         dataset_name, method, str(counter), str(data_len))
    return {'dataset': dataset_name, 'records': records}


test_or_valid = 'only_test'

if __name__ == '__main__':
    initialization()
    model = SimImportanceUnderstanding()
    sim_imp_datasets = []

    for analogy_dataset in configuration.analogy_datasets:
        the_data = read_datasets(analogy_dataset, approach=test_or_valid)
        sim_imp_dataset = analogy_dataset_for_sim_importance_understanding_with_second_appr(the_data,
                                                                                            analogy_dataset, model)
        sim_imp_datasets.append(sim_imp_dataset)

    sim_imp_datasets.append({'relbert': model.relbert_embeddings_for_concept_pairs})
    sim_imp_datasets.append({'composition': model.composition_model_embeddings_for_concept_pairs})
    sim_imp_datasets.append({'related': model.concepts_in_path_length_2})
    dataset_file_name = f'/scratch/c.scmnk4/elexir/resources/sim_importance_datasets_second_appr_' \
                        f'sum_max_min_glove_numberbatch_top5_only_test.pkl'
    # dataset_file_name = f'/scratch/c.scmnk4/elexir/resources/sim_importance_datasets_second_appr_composition_' \
    #                     f'{configuration.inner_layer_dimension}_glove_numberbatch_top5_test_only.pkl'
    # dataset_file_name = f'/scratch/c.scmnk4/elexir/resources/sim_importance_datasets_second_appr_composition_' \
    #                     f'{configuration.inner_layer_dimension}_mean_max_glove_numberbatch_top5.pkl'
    # dataset_file_name = f'/scratch/c.scmnk4/elexir/resources/sim_importance_datasets_second_appr_transformer_' \
    #                     f'{configuration.inner_layer_dimension}_glove_numberbatch_top5.pkl'
    with open(dataset_file_name, 'wb') as f:
        pickle.dump(sim_imp_datasets, f)
