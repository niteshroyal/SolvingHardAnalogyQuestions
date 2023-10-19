import os
import pickle
import logging
import unidecode
from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.core.data_processor import TrainingDataProcessor
from reasoning_with_vectors.importance.importance_filter import Classifier

'''
This file generates data to train Composition Models for the paper.
'''


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] +
                            f'appr_{configuration.dataset_preparation_approch_for_composition_model}.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def write_training_data(data, counter):
    filename = os.path.join(os.path.dirname(configuration.training_dataset),
                            os.path.splitext(os.path.basename(configuration.training_dataset))[0] +
                            '_part_' + str(counter) + '.pkl')
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class DataProcessor(TrainingDataProcessor):
    def __init__(self):
        super().__init__()

    def load_training_concept_pairs(self):
        self.training_concept_pairs = []
        if os.path.isfile(configuration.training_concept_pairs_file):
            logging.info('Target concept pairs file exists so loading target concept pairs from this file')
            with open(configuration.training_concept_pairs_file, 'r') as file:
                data = file.read()
                data = data.split('\n')
                for item in data:
                    if item == '':
                        continue
                    else:
                        item = item.split('\t')
                        self.training_concept_pairs.append(item)
            return True
        else:
            return False

    def write_training_concept_pairs(self):
        with open(configuration.training_concept_pairs_file, 'w') as file:
            for item in self.training_concept_pairs:
                file.write(item[0] + '\t' + item[1] + '\n')

    def is_conceptnet_link(self, c1, c2):
        answer = False
        qry = f'conceptnet_edge("{c1}", "{c2}").'
        for sol in self.prolog.query(qry):
            answer = True
            break
        return answer

    def get_training_concept_pairs_from_db(self):
        if self.load_training_concept_pairs():
            logging.info(f'Number of target concept pairs loaded from {configuration.training_concept_pairs_file} '
                         f'is {len(self.training_concept_pairs)}')
            self.importance_filter = Classifier()
            self.importance_filter.load_model()
        else:
            super().get_training_concept_pairs_from_db()
            self.write_training_concept_pairs()

    def set_path_finder_approach(self):
        answer = False
        qry = f'set_path_finder_approach_for_experiment1' \
              f'("{configuration.dataset_preparation_approch_for_composition_model}")'
        for sol in self.prolog.query(qry):
            answer = True
            break
        return answer

    def get_related_concepts(self, concept1, concept2):
        if configuration.dataset_preparation_approch_for_composition_model in [1.5, 4, 5, 6, 7]:
            topN = configuration.get_related_concepts_using_word_embeddings_topn
        else:
            topN = 0
        rel_concepts = None
        # rel_concepts = self.get_store_related_concepts(concept1, concept2)
        if rel_concepts is not None:
            return rel_concepts
        else:
            rel_concepts = set()
            qry = 'experiment1_iter_path_len_2(' + '"' + concept1 + '","' + concept2 + '",Z)'
            for sol in self.prolog.query(qry):
                z = sol['Z'].decode('UTF-8')
                rel_concepts.add(z)
            concept1_most_similar_via_numberbatch = \
                self.get_glove_similar_concepts(concept1, topN)
            concept2_most_similar_via_numberbatch = \
                self.get_glove_similar_concepts(concept2, topN)
            for concept1_similar in concept1_most_similar_via_numberbatch:
                if '"' in concept1_similar or concept1_similar == '"' or concept1_similar == ' ' or \
                        concept1_similar == '' or concept1_similar == ',' or concept1_similar == '.' or \
                        concept1_similar == "'" or '\\' in concept1_similar or concept1_similar == '/':
                    continue
                qry = 'experiment1_iter_path_len_2(' + '"' + concept1_similar + '","' + concept2 + '",Z)'
                for sol in self.prolog.query(qry):
                    z = sol['Z'].decode('UTF-8')
                    rel_concepts.add(z)
            for concept2_similar in concept2_most_similar_via_numberbatch:
                if '"' in concept2_similar or concept2_similar == '"' or concept2_similar == ' ' or \
                        concept2_similar == '' or concept2_similar == ',' or concept2_similar == '.' \
                        or concept2_similar == "'" or '\\' in concept2_similar or concept2_similar == '/':
                    continue
                qry = 'experiment1_iter_path_len_2(' + '"' + concept1 + '","' + concept2_similar + '",Z)'
                for sol in self.prolog.query(qry):
                    z = sol['Z'].decode('UTF-8')
                    rel_concepts.add(z)
            rel_concepts = list(rel_concepts)
            if concept1 in rel_concepts:
                rel_concepts.remove(concept1)
            if concept2 in rel_concepts:
                rel_concepts.remove(concept2)
            # self.store_related_concepts(concept1, concept2, rel_concepts)
            return rel_concepts

    def extract_training_data(self):
        self.set_path_finder_approach()
        required_relbert_vectors = set()
        for [c1, c2] in self.training_concept_pairs:
            rel_concepts = self.get_related_concepts(c1, c2)
            for item in rel_concepts:
                required_relbert_vectors.add((c1, item))
                required_relbert_vectors.add((item, c2))
        required_relbert_vectors_list = []
        for (c1, c2) in required_relbert_vectors:
            required_relbert_vectors_list.append([c1, c2])
        self.cache_embedding_with_batch_size(required_relbert_vectors_list, configuration.relbert_batch_size)
        counter = 0
        training_data = []
        for [c1, c2] in self.training_concept_pairs:
            rel_concepts = self.get_related_concepts(c1, c2)
            xy = self.get_embedding_for_one_pair([c1, c2])
            xzy = []
            for item in rel_concepts:
                xz = self.get_embedding_for_one_pair([c1, item])
                zy = self.get_embedding_for_one_pair([item, c2])
                if configuration.dataset_preparation_approch_for_composition_model in [1, 1.5, 2, 3, 4, 6, 7]:
                    xzy.append([item, xz, zy])
                elif configuration.dataset_preparation_approch_for_composition_model in [5] and \
                        self.importance_filter.importance(xz) > configuration.importance_threshold and \
                        self.importance_filter.importance(zy) > configuration.importance_threshold:
                    xzy.append([item, xz, zy])
                else:
                    pass
            if len(xzy) == 0:
                continue
            training_data.append([c1, c2, xy, xzy])
            if len(training_data) >= 1000:
                counter += 1
                write_training_data(training_data, counter)
                training_data = []
        if len(training_data) > 0:
            counter += 1
            write_training_data(training_data, counter)


def qualitative_analysis_on_related_concepts():
    obj = DataProcessor()
    concept_pairs_to_analyze = [['blandishment', 'coax'], ['eulogy', 'praise'],
                                ['reprehensible', 'condemn'], ['estimable', 'praise'],
                                ['processing', 'bug'], ['thinking', 'mistake'],
                                ['selection', 'popularity'], ['competition', 'fitness'],
                                ['reflects', 'rough'], ['echoes', 'loud'],
                                ['shore', 'water'], ['wall', 'air'],
                                ['vernacular', 'regional'], ['fluctuation', 'irregular'],
                                ['eccentric', 'codger'], ['admirable', 'hero']]
    for [a, b] in concept_pairs_to_analyze:
        a = unidecode.unidecode(a.lower())
        b = unidecode.unidecode(b.lower())
        print('-----------------------------------------------------')
        print('-----------------------------------------------------')
        for appr in [1, 2, 3, 4]:
            configuration.dataset_preparation_approch_for_composition_model = appr
            obj.set_path_finder_approach()
            rel_concepts = obj.get_related_concepts(a, b)
            print(f'Pair = {[a, b]}, DP_Approach = {appr}, Interim concepts = {rel_concepts}')
            print('-----------------------------------------------------')
            if appr == 4:
                # TODO
                a_ss = []
                b_ss = []
                concept1_most_similar_via_numberbatch = \
                    obj.get_glove_similar_concepts(a, configuration.get_related_concepts_using_word_embeddings_topn)
                for concept1_similar in concept1_most_similar_via_numberbatch:
                    if '"' in concept1_similar or concept1_similar == '"' or concept1_similar == ' ' or \
                            concept1_similar == '' or concept1_similar == ',' or concept1_similar == '.' or \
                            concept1_similar == "'" or '\\' in concept1_similar or concept1_similar == '/':
                        continue
                    qry = 'experiment1_iter_path_len_2(' + '"' + concept1_similar + '","' + b + '",Z)'
                    temp = set()
                    for sol in obj.prolog.query(qry):
                        z = sol['Z'].decode('UTF-8')
                        temp.add(z)
                    a_ss.append((concept1_similar, temp))
                print(f'ss for {a} = {a_ss}')
                print('-----------------------------------------------------')
                concept2_most_similar_via_numberbatch = \
                    obj.get_glove_similar_concepts(b, configuration.get_related_concepts_using_word_embeddings_topn)
                for concept2_similar in concept2_most_similar_via_numberbatch:
                    if '"' in concept2_similar or concept2_similar == '"' or concept2_similar == ' ' or \
                            concept2_similar == '' or concept2_similar == ',' or concept2_similar == '.' \
                            or concept2_similar == "'" or '\\' in concept2_similar or concept2_similar == '/':
                        continue
                    qry = 'experiment1_iter_path_len_2(' + '"' + a + '","' + concept2_similar + '",Z)'
                    temp = set()
                    for sol in obj.prolog.query(qry):
                        z = sol['Z'].decode('UTF-8')
                        temp.add(z)
                    b_ss.append((concept2_similar, temp))
                print(f'ss for {b} = {b_ss}')
                print('-----------------------------------------------------')


if __name__ == '__main__':
    initialization()
    # qualitative_analysis_on_related_concepts()

    obj = DataProcessor()
    obj.get_training_concept_pairs_from_db()
    obj.extract_training_data()
