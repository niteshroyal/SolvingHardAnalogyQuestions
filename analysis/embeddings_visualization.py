import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.utils import shuffle

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.importance.importance_filter_training import Classifier
from reasoning_with_vectors.importance.positive_concept_pairs import part_to_whole, type_and_category, \
    used_to, manner, symbol_or_representation, action_and_significance, degree_of_intensity, synonyms, antonyms, \
    hyponyms, hypernyms, morphologies, spatial_relationships, temporal_relationships, tools_and_materials, \
    sequence_or_hierarchy, syncretic_relationships, complementary_concepts, similes, collocations, \
    agent_and_recipient, coinage_or_neologisms, gender_related, has_property, located_at, shared_features, \
    aesthetic_relationships, co_occurrence_patterns, cognitive_associations, juxtaposition, cause_and_effect, \
    word_families, counterparts, diminutive_and_augmentative_forms, affixation, \
    membership_in_a_common_set, connotative_or_denotative_meanings, associative_relationships, \
    phrasal_verb_concept_pairs, kinship_relationships, sensory_associations, related_concept_pairs

relations = [part_to_whole, type_and_category, used_to, manner, symbol_or_representation, action_and_significance,
             degree_of_intensity, synonyms, antonyms, hyponyms, hypernyms, morphologies, spatial_relationships,
             temporal_relationships, tools_and_materials, sequence_or_hierarchy, syncretic_relationships,
             complementary_concepts, similes, collocations, agent_and_recipient, coinage_or_neologisms, gender_related,
             has_property, located_at, shared_features, aesthetic_relationships, co_occurrence_patterns,
             cognitive_associations, juxtaposition, cause_and_effect, word_families, counterparts,
             diminutive_and_augmentative_forms, affixation, membership_in_a_common_set,
             connotative_or_denotative_meanings, associative_relationships, phrasal_verb_concept_pairs,
             kinship_relationships, sensory_associations]


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)
    logging.info('Started')


def pca_visualization(positive_embedd, negative_embedd, dataset_name):
    logging.info(f'Going to perform UMAP visualization for {dataset_name} concept pairs')
    data = np.vstack((positive_embedd, negative_embedd))
    labels = [f'{dataset_name} concept pairs'] * len(positive_embedd) + \
             ['Random concept pairs'] * len(negative_embedd)

    data, labels = shuffle(data, labels, random_state=42)

    data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

    # # Apply PCA
    # pca = PCA(n_components=2)
    # data_2d = pca.fit_transform(data_normalized)

    # Apply UMAP
    reducer = umap.UMAP(metric='cosine', n_neighbors=4, n_components=2, random_state=42)
    data_2d = reducer.fit_transform(data_normalized)

    plt.figure(figsize=(10, 6))
    colors = {"Random concept pairs": "red", f"{dataset_name} concept pairs": "blue"}
    for label in np.unique(labels):
        x_values = []
        y_values = []
        for idx in range(len(data_2d)):
            if labels[idx] == label:
                x_values.append(data_2d[idx][0])
                y_values.append(data_2d[idx][1])
        plt.scatter(x_values, y_values, label=label, s=4, alpha=0.5, color=colors[label])

    plt.xlim(min(data_2d[:, 0]), max(data_2d[:, 0]))
    plt.ylim(min(data_2d[:, 1]), max(data_2d[:, 1]))

    # plt.legend(fontsize=24)
    plt.xlabel('UMAP Component 1', fontsize=30)
    plt.ylabel('UMAP Component 2', fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.savefig(f'umap_visualization_{dataset_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()


class Visualization(Classifier):
    def __init__(self):
        super().__init__()

    def get_some_conceptnet_edges(self, num):
        conceptnet_links = set()
        qry = 'conceptnet_edge(X,Y).'
        # counter = 0
        for sol in self.extractor.prolog.query(qry):
            c1 = sol['X'].decode('UTF-8')
            c2 = sol['Y'].decode('UTF-8')
            conceptnet_links.add((c1, c2))
            # counter += 1
            # if counter > 100:
            #     break
        conceptnet_links = sorted(list(conceptnet_links))
        rnd = random.Random(88)
        rnd.shuffle(conceptnet_links)
        conceptnet_links_short = []
        counter = 0
        for (a, b) in conceptnet_links:
            conceptnet_links_short.append([a, b])
            counter += 1
            if counter >= num:
                break
        return conceptnet_links_short

    def get_gpt_generated_related_concept_pairs(self):
        unique_related_pairs = set()
        unique_related_pairs_list = []
        for lst in related_concept_pairs + relations:
            for [c1, c2] in lst:
                c1 = str(c1).replace('_', ' ').lower()
                c2 = str(c2).replace('_', ' ').lower()
                unique_related_pairs.add((c1, c2))
        for (c1, c2) in unique_related_pairs:
            unique_related_pairs_list.append([c1, c2])
        self.in_conceptnet = unique_related_pairs_list

    def get_all_gpt4_concept_pairs(self):
        self.get_gpt_generated_related_concept_pairs()
        return self.in_conceptnet

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
        while len(temp) <= positive_examples_size:
            c1 = rnd.choice(concepts_in_conceptnet)
            c2 = rnd.choice(concepts_in_conceptnet)
            if (c1 != c2) and ([c1, c2] not in positive_examples) and \
                    (not self.undirected_link_in_conceptnet(c1, c2)):
                temp.add((c1, c2))
        negative_examples = sorted([[x[0], x[1]] for x in temp], key=lambda x: [x[0], x[1]])
        return negative_examples

    def get_relbert_embeddings(self, concept_pairs):
        embeddings = []
        examples = []
        logging.info(f'Number of concept pairs to obtain RelBERT embeddings = {len(concept_pairs)}')
        for item in concept_pairs:
            if len(examples) > configuration.relbert_batch_size:
                embeddings = embeddings + self.extractor.get_embedding(examples)
                logging.info(f'Number of concept pairs processed = {len(embeddings)}')
                examples = [item]
            else:
                examples.append(item)
        embeddings = embeddings + self.extractor.get_embedding(examples)
        return embeddings

    def visualization(self):
        gpt4_cps = self.get_all_gpt4_concept_pairs()
        logging.info(f'GPT-4 concept pairs = {gpt4_cps}')
        num_of_cps = len(gpt4_cps)
        conceptnet_cps = self.get_some_conceptnet_edges(num_of_cps)
        logging.info(f'ConceptNet pairs = {conceptnet_cps}')
        gpt4_negative_cps = self.generate_negative_examples(gpt4_cps)
        logging.info(f'GPT-4 negative concept pairs = {gpt4_negative_cps}')
        conceptnet_negative_cps = self.generate_negative_examples(conceptnet_cps)
        logging.info(f'ConceptNet negative concept pairs = {conceptnet_negative_cps}')
        gpt4_cps_embedd = self.get_relbert_embeddings(gpt4_cps)
        gpt4_negative_cps_embedd = self.get_relbert_embeddings(gpt4_negative_cps)
        conceptnet_cps_embedd = self.get_relbert_embeddings(conceptnet_cps)
        conceptnet_negative_cps_embedd = self.get_relbert_embeddings(conceptnet_negative_cps)
        pca_visualization(gpt4_cps_embedd, gpt4_negative_cps_embedd, 'GPT-4')
        pca_visualization(conceptnet_cps_embedd, conceptnet_negative_cps_embedd, 'ConceptNet')


if __name__ == '__main__':
    initialization()
    v_obj = Visualization()
    v_obj.visualization()
