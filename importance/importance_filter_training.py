import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.core.data_processor import TrainingDataProcessor


import pickle

from reasoning_with_vectors.importance.positive_concept_pairs import part_to_whole, type_and_category, \
    used_to, manner, symbol_or_representation, action_and_significance, degree_of_intensity, synonyms, antonyms, \
    hyponyms, hypernyms, morphologies, spatial_relationships, temporal_relationships, tools_and_materials, \
    sequence_or_hierarchy, syncretic_relationships, complementary_concepts, cultural_or_historical_associations, \
    similes, collocations, agent_and_recipient, coinage_or_neologisms, gender_related, known_for, has_property, \
    occupation, located_at, capital_of, shared_features, aesthetic_relationships, co_occurrence_patterns, \
    cognitive_associations, phonetic_symbolism, metonymy, allusions_and_references, juxtaposition, cause_and_effect, \
    word_families, counterparts, diminutive_and_augmentative_forms, onomatopoeia, affixation, \
    intensifiers_and_reducers, membership_in_a_common_set, eponymous_relationships, etymological_relationships, \
    connotative_or_denotative_meanings, tautology, quantifiers, abbreviations, reduplication, \
    domain_specific_relationships, measurement_and_scale, conversion_relationships, associative_relationships, \
    phrasal_verb_concept_pairs, loanwords, kinship_relationships, element_and_compound, sensory_associations, \
    related_concept_pairs

# relations = [part_to_whole, type_and_category, used_to, manner, symbol_or_representation, action_and_significance,
#              degree_of_intensity, synonyms, antonyms, hyponyms, hypernyms, homophones, morphologies,
#              spatial_relationships,
#              temporal_relationships, tools_and_materials, sequence_or_hierarchy, syncretic_relationships,
#              complementary_concepts, cultural_or_historical_associations, similes, collocations,
#              metaphorical_relationships,
#              agent_and_recipient, coinage_or_neologisms, gender_related, known_for, has_property, occupation,
#              located_at,
#              capital_of, shared_features, aesthetic_relationships, co_occurrence_patterns, cognitive_associations,
#              phonetic_symbolism, metonymy, allusions_and_references, juxtaposition, acronyms, cause_and_effect,
#              word_families,
#              counterparts, diminutive_and_augmentative_forms, onomatopoeia, affixation, intensifiers_and_reducers,
#              membership_in_a_common_set, eponymous_relationships, etymological_relationships,
#              connotative_or_denotative_meanings, tautology, quantifiers, abbreviations, reduplication,
#              domain_specific_relationships, measurement_and_scale, synaesthetic_metaphor, conversion_relationships,
#              associative_relationships, phrasal_verb_concept_pairs, loanwords, idiomatic_expressions, proverbs,
#              kinship_relationships, element_and_compound, sensory_associations, cliches_and_tropes]

relations = [part_to_whole, type_and_category, used_to, manner, symbol_or_representation, action_and_significance,
             degree_of_intensity, synonyms, antonyms, hyponyms, hypernyms, morphologies,
             spatial_relationships,
             temporal_relationships, tools_and_materials, sequence_or_hierarchy, syncretic_relationships,
             complementary_concepts, cultural_or_historical_associations, similes, collocations,
             agent_and_recipient, coinage_or_neologisms, gender_related, known_for, has_property, occupation,
             located_at,
             capital_of, shared_features, aesthetic_relationships, co_occurrence_patterns, cognitive_associations,
             phonetic_symbolism, metonymy, allusions_and_references, juxtaposition, cause_and_effect,
             word_families,
             counterparts, diminutive_and_augmentative_forms, onomatopoeia, affixation, intensifiers_and_reducers,
             membership_in_a_common_set, eponymous_relationships, etymological_relationships,
             connotative_or_denotative_meanings, tautology, quantifiers, abbreviations, reduplication,
             domain_specific_relationships, measurement_and_scale, conversion_relationships,
             associative_relationships, phrasal_verb_concept_pairs, loanwords,
             kinship_relationships, element_and_compound, sensory_associations]


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)
    logging.info('Started')


def cosine_kernel(X, Y):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return X_normalized @ Y_normalized.T


class Classifier:
    def __init__(self):
        self.extractor = TrainingDataProcessor()
        self.in_conceptnet = []
        self.not_in_conceptnet = []
        self.positive_examples = []
        self.negative_examples = []
        self.model = None

    def undirected_link_in_conceptnet(self, c1, c2):
        present_in_conceptnet = False
        qry = 'undirected_link(' + '"' + c1 + '", ' + '"' + c2 + '"' + ').'
        for sol in self.extractor.prolog.query(qry):
            present_in_conceptnet = True
        return present_in_conceptnet

    def get_gpt_generated_related_concept_pairs(self):
        unique_related_pairs = set()
        unique_related_pairs_list = []
        for lst in related_concept_pairs + relations:
            for [c1, c2] in lst:
                c1 = str(c1).replace('_', ' ').lower()
                c2 = str(c2).replace('_', ' ').lower()
                unique_related_pairs.add((c1, c2))
                unique_related_pairs.add((c2, c1))
        for (c1, c2) in unique_related_pairs:
            unique_related_pairs_list.append([c1, c2])
        self.in_conceptnet = unique_related_pairs_list

    def validation_by_concept_pair_inverse(self):
        threshold = 0.5
        validation_set = []
        for [c1, c2] in self.in_conceptnet:
            if [c2, c1] in self.in_conceptnet:
                pass
            else:
                validation_set.append([c2, c1])

        positive = self.get_stored_relbert_embds_1(validation_set)

        data = np.vstack(positive)
        labels = [1] * len(positive)

        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
        y_pred = []
        probs = self.model.predict_proba(data_normalized)
        for [_, p] in probs:
            if p > threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        accuracy = accuracy_score(labels, y_pred)
        print(f"Validation Concept Pair Inverse Accuracy: {accuracy:.4f}")
        logging.info('Validation Concept Pair Inverse Accuracy = %s', str(accuracy))

    def get_stored_relbert_embds_1(self, keys):
        result = []
        for key in keys:
            result.append(self.extractor.get_stored_relbert_embds(key))
        return result

    def generate_negative_examples(self, seed=42):
        self.positive_examples = self.in_conceptnet
        positive_examples_size = len(self.positive_examples)
        concepts_in_conceptnet = set()
        for item in self.positive_examples:
            [c1, c2] = item
            concepts_in_conceptnet.add(c1)
            concepts_in_conceptnet.add(c2)
        concepts_in_conceptnet = sorted(list(concepts_in_conceptnet))

        rnd = random.Random(seed)

        temp = set()
        while len(temp) <= positive_examples_size:
            c1 = rnd.choice(concepts_in_conceptnet)
            c2 = rnd.choice(concepts_in_conceptnet)
            if (c1 != c2) and ([c1, c2] not in self.positive_examples) and \
                    (not self.undirected_link_in_conceptnet(c1, c2)):
                temp.add((c1, c2))
        self.negative_examples = sorted([[x[0], x[1]] for x in temp], key=lambda x: [x[0], x[1]])

    def cache_positive_example(self):
        examples = []
        logging.info('%s number of positive datapoints to cache', str(len(self.positive_examples)))
        for item in self.positive_examples:
            if len(examples) > configuration.relbert_batch_size:
                self.extractor.get_embedding(examples)
                logging.info('%s number of datapoints cached', str(configuration.relbert_batch_size))
                examples = [item]
            else:
                examples.append(item)
        self.extractor.get_embedding(examples)

    def cache_negative_example(self):
        examples = []
        logging.info('%s number of negative datapoints to cache', str(len(self.negative_examples)))
        for item in self.negative_examples:
            if len(examples) > configuration.relbert_batch_size:
                self.extractor.get_embedding(examples)
                logging.info('%s number of datapoints cached', str(configuration.relbert_batch_size))
                examples = [item]
            else:
                examples.append(item)
        self.extractor.get_embedding(examples)

    def sort_examples(self):
        positive = self.get_stored_relbert_embds_1(self.positive_examples)
        negative = self.get_stored_relbert_embds_1(self.negative_examples)

        positive_negative_examples = self.positive_examples + self.negative_examples

        # Combine both lists into a single dataset and create corresponding labels
        trainingdata = np.vstack((positive, negative))
        labels = [1] * len(positive) + [0] * len(negative)

        trainingdata = trainingdata / np.linalg.norm(trainingdata, axis=1, keepdims=True)

        self.load_model()

        # Predict probabilities for the test set
        probabilities = self.model.predict_proba(trainingdata)

        # Combine the test data, true labels, and predicted probabilities into a single DataFrame
        results_df = pd.DataFrame(data=positive_negative_examples, columns=['Concept1', 'Concept2'])
        results_df['True_Label'] = labels
        results_df['Positive_Class_Prob'] = probabilities[:, 1]
        results_df['Negative_Class_Prob'] = probabilities[:, 0]

        # Sort positive and negative examples separately
        positive_examples_sorted = results_df[results_df['True_Label'] == 1].sort_values(by='Positive_Class_Prob',
                                                                                         ascending=False)
        negative_examples_sorted = results_df[results_df['True_Label'] == 0].sort_values(by='Negative_Class_Prob',
                                                                                         ascending=False)

        # Save sorted examples to separate files
        positive_examples_sorted.to_csv('positive_examples_sorted.csv', index=False)
        negative_examples_sorted.to_csv('negative_examples_sorted.csv', index=False)

    def load_model(self):
        with open(configuration.classifier_model_path, 'rb') as f:
            self.model = pickle.load(f)

    def importance1(self, data):
        data = np.vstack(data)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        probabilities = self.model.predict_proba(data)
        importance = []
        for [_, b] in probabilities:
            importance.append(b)
        return importance

    def importance(self, data):
        if type(data[0]) is list:
            importance = self.importance1(data)
        else:
            importance = self.importance1([data])[0]
        return importance

    def xgboost_classifier_test(self):
        positive = self.get_stored_relbert_embds_1(self.positive_examples)
        negative = self.get_stored_relbert_embds_1(self.negative_examples)

        # Combine both lists into a single dataset and create corresponding labels
        data = np.vstack((positive, negative))
        labels = [1] * len(positive) + [0] * len(negative)

        data, labels = shuffle(data, labels, random_state=42)
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.1, random_state=42)

        # Train the XGBoost model
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.model.fit(X_train, y_train)

        # Save the XGBoost model to a file
        with open(configuration.classifier_model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Compute training accuracy
        y_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        print(f"Train Accuracy: {accuracy:.4f}")
        logging.info('Xgboost Train Accuracy = %s', str(accuracy))

        self.validation_by_concept_pair_inverse()

        # Compute test accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        logging.info('Xgboost Test Accuracy = %s', str(accuracy))

    def simple_logistic_regression_classifier_test(self):
        positive = self.get_stored_relbert_embds_1(self.positive_examples)
        negative = self.get_stored_relbert_embds_1(self.negative_examples)

        # Combine both lists into a single dataset and create corresponding labels
        data = np.vstack((positive, negative))
        labels = [1] * len(positive) + [0] * len(negative)

        data, labels = shuffle(data, labels, random_state=42)
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.1, random_state=42)

        # Train the logistic regression model
        self.model = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
        self.model.fit(data_normalized, labels)

        # Save the logistic regression model to a file
        with open(configuration.classifier_model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Compute training accuracy
        y_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        print(f"Train Accuracy: {accuracy:.4f}")
        logging.info('Logistic Regression Train Accuracy = %s', str(accuracy))

        self.validation_by_concept_pair_inverse()

        # Compute test accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        logging.info('Logistic Regression Test Accuracy = %s', str(accuracy))

    def simple_logistic_regression_classifier(self):
        positive = self.get_stored_relbert_embds_1(self.positive_examples)
        negative = self.get_stored_relbert_embds_1(self.negative_examples)

        # Combine both lists into a single dataset and create corresponding labels
        data = np.vstack((positive, negative))
        labels = [1] * len(positive) + [0] * len(negative)

        data, labels = shuffle(data, labels, random_state=42)
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Train the logistic regression model
        self.model = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
        self.model.fit(data_normalized, labels)

        # Save the logistic regression model to a file
        with open(configuration.classifier_model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Make predictions and compute accuracy
        y_pred = self.model.predict(data_normalized)
        accuracy = accuracy_score(labels, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        logging.info('simple_logistic_regression_classifier accuracy = %s', str(accuracy))

    def svm_rbf_classifier_test(self):
        positive = self.get_stored_relbert_embds_1(self.positive_examples)
        negative = self.get_stored_relbert_embds_1(self.negative_examples)

        # Combine both lists into a single dataset and create corresponding labels
        data = np.vstack((positive, negative))
        labels = [1] * len(positive) + [0] * len(negative)

        data, labels = shuffle(data, labels, random_state=42)
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.1, random_state=42)

        # Train the SVM with RBF kernel
        self.model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        self.model.fit(X_train, y_train)

        # Save the SVM with RBF kernel model to a file
        with open(configuration.classifier_model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Compute training accuracy
        y_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        print(f"Train Accuracy: {accuracy:.4f}")
        logging.info('Svm with Rbf Train Accuracy = %s', str(accuracy))

        self.validation_by_concept_pair_inverse()

        # Compute test accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        logging.info('Svm with Rbf Test Accuracy = %s', str(accuracy))

    def svm_rbf_classifier(self):
        positive = self.get_stored_relbert_embds_1(self.positive_examples)
        negative = self.get_stored_relbert_embds_1(self.negative_examples)

        # Combine both lists into a single dataset and create corresponding labels
        data = np.vstack((positive, negative))
        labels = [1] * len(positive) + [0] * len(negative)

        data, labels = shuffle(data, labels, random_state=42)
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Train the SVM with RBF kernel
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        self.model.fit(data_normalized, labels)

        # Save the SVM with RBF kernel model to a file
        with open(configuration.classifier_model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Make predictions and compute accuracy
        y_pred = self.model.predict(data_normalized)
        accuracy = accuracy_score(labels, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        logging.info('simple_svm_rbf_classifier accuracy = %s', str(accuracy))

    def scatter_plot(self):
        in_conceptnet = self.get_stored_relbert_embds_1(self.positive_examples)
        not_in_conceptnet = self.get_stored_relbert_embds_1(self.negative_examples)

        # Combine both lists into a single dataset and create corresponding labels
        data = np.vstack((in_conceptnet, not_in_conceptnet))
        labels = ['positive relations'] * len(in_conceptnet) + ['negative relations'] * len(not_in_conceptnet)

        data, labels = shuffle(data, labels, random_state=42)

        # # Apply t-SNE
        # tsne = TSNE(n_components=2, random_state=42)
        # data_2d = tsne.fit_transform(data)

        # Normalize each data point to a unit vector
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Apply PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_normalized)

        # # Apply UMAP
        # reducer = umap.UMAP(metric='cosine', n_neighbors=15, n_components=2, random_state=42)
        # data_2d = reducer.fit_transform(data)

        # # Apply MDS
        # cosine_similarity_matrix = 1 - pairwise_distances(data, metric='cosine')
        # mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        # data_2d = mds.fit_transform(cosine_similarity_matrix)

        # Create the plot
        plt.figure(figsize=(10, 6))
        for label in np.unique(labels):
            x_values = []
            y_values = []
            for idx in range(len(data_2d)):
                if labels[idx] == label:
                    x_values.append(data_2d[idx][0])
                    y_values.append(data_2d[idx][1])
            plt.scatter(x_values, y_values, label=label, s=2, alpha=0.5)

        # Set the axis limits based on the transformed data
        plt.xlim(min(data_2d[:, 0]), max(data_2d[:, 0]))
        plt.ylim(min(data_2d[:, 1]), max(data_2d[:, 1]))

        plt.legend()
        # plt.xlabel('t-SNE Component 1')
        # plt.ylabel('t-SNE Component 2')
        # plt.title('t-SNE Visualization of RelBERT Vectors')

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA Visualization of RelBERT Vectors')

        # plt.xlabel('UMAP 1')
        # plt.ylabel('UMAP 2')
        # plt.title('UMAP Visualization of RelBERT Vectors')

        # plt.xlabel('MDS 1')
        # plt.ylabel('MDS 2')
        # plt.title('MDS Visualization of RelBERT Vectors')

        # plt.show()

        # Save the plot as a PDF file
        # plt.savefig('tsne_visualization.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('pca_visualization_normalized2.pdf', format='pdf', bbox_inches='tight')
        # plt.savefig('umap_visualization.pdf', format='pdf', bbox_inches='tight')
        # plt.savefig('mds_visualization.pdf', format='pdf', bbox_inches='tight')

        # Close the plot to free up resources
        plt.close()


if __name__ == '__main__':
    initialization()
    classifier = Classifier()

    # Train
    classifier.get_gpt_generated_related_concept_pairs()
    classifier.generate_negative_examples()
    classifier.cache_positive_example()
    classifier.cache_negative_example()
    classifier.simple_logistic_regression_classifier()

    # # Visualize
    # classifier.sort_examples()
    # classifier.scatter_plot()

    # Predict
    # datapoints = classifier.extractor.get_embedding([['mars', 'horse'], ['love', 'heart'],
    #                                                  ['timing', 'cup'], ['mind', 'peace'],
    #                                                  ['heart', 'love'], ['peace', 'mind'],
    #                                                  ['smoking', 'lung cancer'], ['lung cancer', 'smoking'],
    #                                                  ['sleep', 'depression'], ['depression', 'sleep'],
    #                                                  ['obesity', 'diabetes'], ['diabetes', 'obesity']])
    # datapoints = classifier.extractor.get_embedding([['diversity', 'resilience'],
    #                                                  ['resilience', 'diversity'],
    #                                                  ['gentrification', 'displacement'],
    #                                                  ['displacement', 'gentrification'],
    #                                                  ['education', 'empowerment'],
    #                                                  ['empowerment', 'education'],
    #                                                  ['natural selection', 'evolution'],
    #                                                  ['evolution', 'natural selection'],
    #                                                  ['broken pipe', 'water leak'],
    #                                                  ['water leak', 'broken pipe']])
    # compressor = Compressor(extractor=classifier.extractor)
    # compressor.load_model()
    # compressed_datapoints = compressor.encode(datapoints)
    # tranformed_datapoints = compressor.decode(compressed_datapoints)
    #
    # classifier.load_model()
    # probs = classifier.importance(datapoints)
    # print(probs)

    # classifier.get_gpt_generated_related_concept_pairs()
    # classifier.load_model()
    # classifier.validation_by_concept_pair_inverse()
