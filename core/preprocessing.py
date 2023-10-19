import logging
import unidecode
import gensim.downloader

from pyswip import Prolog
from pyswip.core import *

from reasoning_with_vectors.conf import configuration
# from reasoning_with_vectors.core.cache import Cache
from reasoning_with_vectors.core.lmdb_cache import LMDB


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


# class Processor(Cache):
#     def __init__(self, read_only=False, db_path=configuration.rocksdb_path, lock_path=configuration.rocksdb_lock):

class Processor(LMDB):
    def __init__(self, read_only=False, db_path=configuration.lmdb_path, lock_path=None):
        super().__init__(read_only, db_path, lock_path)
        self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
        self.prolog = Prolog()
        self.prolog.consult(configuration.knowledge_graph_file)
        # self.numberbatch_vectors = gensim.models.KeyedVectors. \
        #     load_word2vec_format('/scratch/c.scmnk4/elexir/resources/numberbatch-en-19.08.txt',
        #                          binary=False, unicode_errors='ignore')

    def get_numberbatch_similar_concepts(self, concept):
        most_similar_concepts = []
        topn = configuration.get_related_concepts_using_word_embeddings_topn
        if topn == 0:
            return []
        try:
            temp = self.numberbatch_vectors.most_similar(positive=[concept], topn=topn)
        except KeyError:
            temp = []
        for (item, sim) in temp:
            item = unidecode.unidecode(item)
            most_similar_concepts.append(item.replace('_', ' '))
        return most_similar_concepts

    def get_glove_similar_concepts(self, concept, topn=configuration.get_related_concepts_using_word_embeddings_topn):
        most_similar_concepts = []
        if topn == 0:
            return []
        try:
            temp = self.glove_vectors.most_similar(positive=[concept], topn=topn)
        except KeyError:
            temp = []
        for (item, sim) in temp:
            item = unidecode.unidecode(item)
            most_similar_concepts.append(item.replace('_', ' '))
        return most_similar_concepts

    def get_related_concepts(self, concept1, concept2):
        rel_concepts = self.get_store_related_concepts(concept1, concept2)
        if rel_concepts is not None:
            return rel_concepts
        else:
            rel_concepts = set()
            qry = 'iter_path_len_2(' + '"' + concept1 + '","' + concept2 + '",Z)'
            for sol in self.prolog.query(qry):
                z = sol['Z'].decode('UTF-8')
                rel_concepts.add(z)
            # concept1_most_similar_via_numberbatch = self.get_numberbatch_similar_concepts(concept1)
            # concept2_most_similar_via_numberbatch = self.get_numberbatch_similar_concepts(concept2)
            concept1_most_similar_via_numberbatch = self.get_glove_similar_concepts(concept1)
            concept2_most_similar_via_numberbatch = self.get_glove_similar_concepts(concept2)
            for concept1_similar in concept1_most_similar_via_numberbatch:
                if '"' in concept1_similar or concept1_similar == '"' or concept1_similar == ' ' or \
                        concept1_similar == '' or concept1_similar == ',' or concept1_similar == '.' or \
                        concept1_similar == "'" or '\\' in concept1_similar or concept1_similar == '/':
                    continue
                qry = 'iter_path_len_2(' + '"' + concept1_similar + '","' + concept2 + '",Z)'
                for sol in self.prolog.query(qry):
                    z = sol['Z'].decode('UTF-8')
                    rel_concepts.add(z)
            for concept2_similar in concept2_most_similar_via_numberbatch:
                if '"' in concept2_similar or concept2_similar == '"' or concept2_similar == ' ' or \
                        concept2_similar == '' or concept2_similar == ',' or concept2_similar == '.' \
                        or concept2_similar == "'" or '\\' in concept2_similar or concept2_similar == '/':
                    continue
                qry = 'iter_path_len_2(' + '"' + concept1 + '","' + concept2_similar + '",Z)'
                for sol in self.prolog.query(qry):
                    z = sol['Z'].decode('UTF-8')
                    rel_concepts.add(z)
            rel_concepts = list(rel_concepts)
            if concept1 in rel_concepts:
                rel_concepts.remove(concept1)
            if concept2 in rel_concepts:
                rel_concepts.remove(concept2)
            self.store_related_concepts(concept1, concept2, rel_concepts)
            return rel_concepts

    def get_concept_pairs_from_path_length_3(self, concept1, concept2):
        concept_pairs = self.get_store_concept_pairs_path_length_3(concept1, concept2)
        if concept_pairs is not None:
            return concept_pairs
        else:
            concept_pairs = []
            qry = 'iter_path_len_3(' + '"' + concept1 + '", "' + concept2 + '", B, C)'
            for sol in self.prolog.query(qry):
                concept_pairs.append([sol['B'].decode('UTF-8'), sol['C'].decode('UTF-8')])
            self.store_concept_pairs_path_length_3(concept1, concept2, concept_pairs)
            return concept_pairs

    def store_relbert_vectors_required_for_evaluation(self):
        cooccurs = self.extract_cooccurs()
        logging.info('%s number of cooccurs to process.', str(len(cooccurs)))
        set_to_process = set()
        for (c1, c2) in cooccurs:
            set_to_process.add((c1, c2))
            set_to_process.add((c2, c1))
            rel_concepts = self.get_related_concepts(c1, c2)
            logging.info(f'Concept1: {c1}, Concept2: {c2}, Related_Concepts: {rel_concepts}')
            for item in rel_concepts:
                set_to_process.add((c1, item))
                set_to_process.add((item, c1))
                set_to_process.add((c2, item))
                set_to_process.add((item, c2))
        list_to_process = []
        for (a, b) in set_to_process:
            list_to_process.append([a, b])
        self.cache_embedding_with_batch_size(list_to_process, configuration.relbert_batch_size)

    def extract_cooccurs(self):
        cooccurs = []
        qry = 'concept_pairs(X,Y)'
        for sol in self.prolog.query(qry):
            c1 = sol['X'].decode('UTF-8')
            c2 = sol['Y'].decode('UTF-8')
            cooccurs.append([c1, c2])
        return cooccurs


if __name__ == '__main__':
    initialization()
    obj = Processor()
    obj.store_relbert_vectors_required_for_evaluation()
    # print(obj.get_related_concepts('dog', 'puppy'))
