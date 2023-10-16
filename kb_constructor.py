import logging

import gensim.downloader
import unidecode
from pyswip import Prolog
from pyswip.core import *

from pythonProject.research.reasoning_with_vectors.conf import configuration
from cache import CustomRelBERT
from importance_filter import Classifier


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


topn_similar = 250

header = f'''%%% -*- Mode: Prolog; -*-
% This file consists of relations in Glove and Numberbatch. Vocabulary was restricted to concepts in 
% {configuration.relative_init_file}. For each concept approx. {topn_similar} most similar concepts according to 
% Golve and Numberbatch embeddings was determined. These concept pairs are represented using the relation r/2.   

'''

header_of_filter_file = f'''%%% -*- Mode: Prolog; -*-
% This file consists of filtered relations in {configuration.glove_file}. Each relation, i.e., r/2 has importance 
% greater than {configuration.importance_filter_threshold} according to the importance filter.
 
'''


class RelationEmbedding:
    def __init__(self):
        self.importance_filter = Classifier()
        self.importance_filter.load_model()
        self.model = CustomRelBERT("relbert/relbert-roberta-large-nce-semeval2012-0-400")
        self.filtered_kb_file_handler = None
        self.glove_vectors = None
        self.numberbatch_vectors = None
        self.vocab = None

    def create_dictionary(self):
        prolog = Prolog()
        prolog.consult(configuration.relative_init_file)
        concepts = set()
        qry = 'considered_concepts(X)'
        for sol in prolog.query(qry):
            concept = sol['X']
            concepts.add(concept.decode('UTF-8').replace(' ', '_'))
        self.vocab = []
        self.vocab = list(sorted(concepts))

    def init_concept_vectors(self):
        self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
        self.numberbatch_vectors = gensim.models.KeyedVectors. \
            load_word2vec_format('/scratch/c.scmnk4/elexir/resources/numberbatch-en-19.08.txt',
                                 binary=False, unicode_errors='ignore')
        self.create_dictionary()

    def get_similar_concepts(self, concept):
        golve_similar_concepts = []
        if concept in self.glove_vectors.key_to_index:
            golve_similar_concepts = self.glove_vectors.most_similar(positive=[concept], topn=topn_similar)
            golve_similar_concepts = [c for (c, _) in golve_similar_concepts if c in self.vocab]
        numberbatch_similar_concepts = []
        if concept in self.numberbatch_vectors.key_to_index:
            numberbatch_similar_concepts = self.numberbatch_vectors.most_similar(positive=[concept], topn=topn_similar)
            numberbatch_similar_concepts = [c for (c, _) in numberbatch_similar_concepts if c in self.vocab]
        similar_concepts = set(golve_similar_concepts + numberbatch_similar_concepts)
        return similar_concepts

    def glove_kb(self):
        self.init_concept_vectors()
        concept_pairs = set()
        logging.info(f'Going to find {topn_similar} similar concepts for {len(self.vocab)} concepts')
        counter = 0
        for concept1 in self.vocab:
            similar_concepts = self.get_similar_concepts(concept1)
            for concept2 in similar_concepts:
                if concept1 != concept2:
                    concept_pairs.add((concept1, concept2))
            counter += 1
            if counter % 100 == 0:
                logging.info(f'Number of concepts processed = {counter}')
        file_handler = open(configuration.glove_file, 'w')
        file_handler.write(header)
        for (c1, c2) in concept_pairs:
            if '"' in c1 or '"' in c2:
                continue
            c1 = unidecode.unidecode(c1)
            c1 = c1.replace('_', ' ').lower()
            c2 = unidecode.unidecode(c2)
            c2 = c2.replace('_', ' ').lower()
            file_handler.write('r("' + str(c1) + '", "' + str(c2) + '").\n')
        file_handler.close()

    def relbert_embeddings_no_cache(self, list_of_pairs):
        return self.model.get_embedding(list_of_pairs)

    def filter(self, links):
        links_list = []
        for c1, c2 in links:
            links_list.append([c1, c2])
        embeddings = self.relbert_embeddings_no_cache(links_list)
        scores = self.importance_filter.importance(embeddings)
        for i in range(0, len(links_list)):
            if scores[i] > configuration.importance_filter_threshold:
                [c1, c2] = links_list[i]
                self.filtered_kb_file_handler.write('r("' + str(c1) + '", "' + str(c2) + '").\n')

    def construct_new_kb(self):
        prolog = Prolog()
        prolog.consult(configuration.glove_file)
        self.filtered_kb_file_handler = open(configuration.glove_filtered_file, 'w')
        self.filtered_kb_file_handler.write(header)
        links = set()
        qry = 'r(X,Y).'
        counter = 0
        for sol in prolog.query(qry):
            c1 = sol['X'].decode('UTF-8')
            c2 = sol['Y'].decode('UTF-8')
            links.add((c1, c2))
            if len(links) >= configuration.relbert_batch_size:
                self.filter(links)
                links = set()
            counter += 1
            if counter % 10000 == 0:
                logging.info('Number of links processed = %d', counter)
        if len(links) > 0:
            self.filter(links)
        self.filtered_kb_file_handler.close()


if __name__ == '__main__':
    initialization()
    obj = RelationEmbedding()
    obj.glove_kb()
    obj.construct_new_kb()
