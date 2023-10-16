import os
import logging
import random
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.core.data_processor import TrainingDataProcessor

from reasoning_with_vectors.importance.positive_concept_pairs import part_to_whole, type_and_category, \
    used_to, manner, symbol_or_representation, action_and_significance, degree_of_intensity, synonyms, antonyms, \
    hyponyms, hypernyms, morphologies, spatial_relationships, temporal_relationships, tools_and_materials, \
    sequence_or_hierarchy, syncretic_relationships, complementary_concepts, similes, collocations, \
    agent_and_recipient, coinage_or_neologisms, gender_related, has_property, located_at, shared_features, \
    aesthetic_relationships, co_occurrence_patterns, cognitive_associations, juxtaposition, cause_and_effect, \
    word_families, counterparts, diminutive_and_augmentative_forms, affixation, \
    membership_in_a_common_set, connotative_or_denotative_meanings, associative_relationships, \
    phrasal_verb_concept_pairs, kinship_relationships, sensory_associations, related_concept_pairs, \
    extra_related_concept_pairs, extra_extra_related_concept_pairs, extra_extra_extra_related_concept_pairs, \
    extra_extra_extra_extra_related_concept_pairs, extra_extra_extra_extra_extra_related_concept_pairs

relations = [part_to_whole, type_and_category, used_to, manner, symbol_or_representation, action_and_significance,
             degree_of_intensity, synonyms, antonyms, hyponyms, hypernyms, morphologies, spatial_relationships,
             temporal_relationships, tools_and_materials, sequence_or_hierarchy, syncretic_relationships,
             complementary_concepts, similes, collocations, agent_and_recipient, coinage_or_neologisms, gender_related,
             has_property, located_at, shared_features, aesthetic_relationships, co_occurrence_patterns,
             cognitive_associations, juxtaposition, cause_and_effect, word_families, counterparts,
             diminutive_and_augmentative_forms, affixation, membership_in_a_common_set,
             connotative_or_denotative_meanings, associative_relationships, phrasal_verb_concept_pairs,
             kinship_relationships, sensory_associations, extra_related_concept_pairs,
             extra_extra_related_concept_pairs, extra_extra_extra_related_concept_pairs,
             extra_extra_extra_extra_related_concept_pairs, extra_extra_extra_extra_extra_related_concept_pairs]


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)
    logging.info('Started')


def highly_quality_concept_pairs_from_gpt4_print():
    unique_related_pairs = set()
    unique_related_pairs_list = []
    for lst in related_concept_pairs + relations:
        for [c1, c2] in lst:
            c1 = str(c1).replace('_', ' ').lower()
            c2 = str(c2).replace('_', ' ').lower()
            unique_related_pairs.add((c1, c2))
            # unique_related_pairs.add((c2, c1))
    for (c1, c2) in unique_related_pairs:
        unique_related_pairs_list.append([c1, c2])
    highly_quality_concept_pairs = unique_related_pairs_list
    embeddings = []
    examples = []
    logging.info(f'Number of positive concept pairs to obtain RelBERT embeddings = '
                 f'{len(highly_quality_concept_pairs)}')
    # for item in highly_quality_concept_pairs:
    #     if len(examples) > configuration.relbert_batch_size:
    #         embeddings = embeddings + self.get_embedding(examples)
    #         logging.info(f'Number of concept pairs processed = {len(embeddings)}')
    #         examples = [item]
    #     else:
    #         examples.append(item)
    # embeddings = embeddings + self.get_embedding(examples)

    with open('positive_concept_pairs.txt', 'w') as f:
        for [aa, bb] in highly_quality_concept_pairs:
            f.write(f'positive("{aa}", "{bb}")\n')

    return highly_quality_concept_pairs


class ImportanceClassifierTraining(TrainingDataProcessor):
    def __init__(self):
        super().__init__()
        self.target_concept_pairs = []
        self.training_embeddings = []
        self.classification_model = None

    def is_conceptnet_link(self, c1, c2):
        answer = False
        qry = f'conceptnet_edge("{c1}", "{c2}").'
        for sol in self.prolog.query(qry):
            answer = True
            break
        return answer

    def undirected_link_in_conceptnet(self, c1, c2):
        present_in_conceptnet = False
        qry = 'undirected_link(' + '"' + c1 + '", ' + '"' + c2 + '"' + ').'
        for sol in self.prolog.query(qry):
            present_in_conceptnet = True
            break
        return present_in_conceptnet

    def generate_negative_examples(self, positive_examples, seed=42):
        positive_examples_size = len(positive_examples)
        concepts_in_conceptnet = set()
        for item in positive_examples:
            [c1, c2] = item
            concepts_in_conceptnet.add(c1)
            concepts_in_conceptnet.add(c2)
        concepts_in_conceptnet = sorted(list(concepts_in_conceptnet))
        rnd = random.Random(seed)
        temp = set()
        negative_examples_size = 1 * positive_examples_size
        while len(temp) <= negative_examples_size:
            c1 = rnd.choice(concepts_in_conceptnet)
            c2 = rnd.choice(concepts_in_conceptnet)
            if (c1 != c2) and ([c1, c2] not in positive_examples) and \
                    (not self.undirected_link_in_conceptnet(c1, c2)):
                temp.add((c1, c2))
        negative_examples = sorted([[x[0], x[1]] for x in temp], key=lambda x: [x[0], x[1]])
        with open('negative_concept_pairs.txt', 'w') as f:
            for [aa, bb] in negative_examples:
                f.write(f'negative("{aa}", "{bb}")\n')
        return negative_examples

    def get_relbert_embeddings_for_negative_examples(self, concept_pairs):
        embeddings = []
        examples = []
        logging.info(f'Number of negative concept pairs to obtain RelBERT embeddings = {len(concept_pairs)}')
        for item in concept_pairs:
            if len(examples) > configuration.relbert_batch_size:
                embeddings = embeddings + self.model.get_embedding(examples)
                logging.info(f'Number of concept pairs processed = {len(embeddings)}')
                examples = [item]
            else:
                examples.append(item)
        embeddings = embeddings + self.model.get_embedding(examples)
        return embeddings

    def highly_quality_concept_pairs_from_gpt4(self):
        unique_related_pairs = set()
        unique_related_pairs_list = []
        for lst in related_concept_pairs + relations:
            for [c1, c2] in lst:
                c1 = str(c1).replace('_', ' ').lower()
                c2 = str(c2).replace('_', ' ').lower()
                unique_related_pairs.add((c1, c2))
                # unique_related_pairs.add((c2, c1))
        for (c1, c2) in unique_related_pairs:
            unique_related_pairs_list.append([c1, c2])
        highly_quality_concept_pairs = unique_related_pairs_list
        embeddings = []
        examples = []
        logging.info(f'Number of positive concept pairs to obtain RelBERT embeddings = '
                     f'{len(highly_quality_concept_pairs)}')
        for item in highly_quality_concept_pairs:
            if len(examples) > configuration.relbert_batch_size:
                embeddings = embeddings + self.get_embedding(examples)
                logging.info(f'Number of concept pairs processed = {len(embeddings)}')
                examples = [item]
            else:
                examples.append(item)
        embeddings = embeddings + self.get_embedding(examples)
        return [embeddings, highly_quality_concept_pairs]

    def get_training_concept_pairs_from_db(self):
        logging.info(f'Going to extract ConceptNet concept pairs from {configuration.lmdb_path}')
        self.training_embeddings = []
        self.target_concept_pairs = []
        counter = 0
        for concept_pair, embedding in self.iter_stored_relbert_embds():
            counter += 1
            [c1, c2] = concept_pair
            if self.is_conceptnet_link(c1, c2) and 3 < len(c1) and 3 < len(c2) and c1 != c2:
                self.target_concept_pairs.append(concept_pair)
                self.training_embeddings.append(embedding)
            if counter % 10000 == 0:
                logging.info(f'Number of DB links processed = {counter}')
                logging.info(f'Number of ConceptNet concept pairs extracted = {len(self.target_concept_pairs)}')
            if len(self.target_concept_pairs) >= 100000:
                break
        logging.info(f'Total number of ConceptNet concept pairs extracted = {len(self.target_concept_pairs)}')
        logging.info(f'Extracted concept pairs = {self.target_concept_pairs}')
        return [self.training_embeddings, self.target_concept_pairs]

    def simple_logistic_regression_classifier(self, positive, negative):

        # Combine both lists into a single dataset and create corresponding labels
        data = np.vstack((positive, negative))
        labels = [1] * len(positive) + [0] * len(negative)

        data, labels = shuffle(data, labels, random_state=42)
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Train the logistic regression model
        self.classification_model = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
        self.classification_model.fit(data_normalized, labels)

        # Save the logistic regression model to a file
        with open(classifier_model_path, 'wb') as f:
            pickle.dump(self.classification_model, f)

        # Make predictions and compute accuracy
        y_pred = self.classification_model.predict(data_normalized)
        accuracy = accuracy_score(labels, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        logging.info('simple_logistic_regression_classifier accuracy = %s', str(accuracy))


if __name__ == '__main__':
    # classifier_model_path = '/scratch/c.scmnk4/elexir/resources/learned_models/' \
    #                         'gpt4_high_quality_importance_classifier.pkl'
    # initialization()
    obj = ImportanceClassifierTraining()
    a = highly_quality_concept_pairs_from_gpt4_print()
    obj.generate_negative_examples(a)

    # # [a, b] = obj.get_training_concept_pairs_from_db()
    # [a, b] = obj.highly_quality_concept_pairs_from_gpt4()
    # c = obj.generate_negative_examples(b)
    # d = obj.get_relbert_embeddings_for_negative_examples(c)
    # obj.simple_logistic_regression_classifier(a, d)
