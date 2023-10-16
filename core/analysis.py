import re
import ast
import pickle
import unidecode
import numpy as np
from sklearn.preprocessing import normalize

from relbert import cosine_similarity

from reasoning_with_vectors.importance.importance_filter import Classifier


def calculate_accuracy(result):
    size = len(result)
    if size == 0:
        return [0, 0]
    correct = 0
    for item in result:
        if item:
            correct += 1
        else:
            pass
    return [size, (correct / float(size)) * 100]


def get_lookup_key(dataset, stem, choices):
    query = '_'.join(stem)
    choices = '__'.join('_'.join(choice) for choice in choices)
    key = 'dataset_{}__query__{}__choices__{}'.format(dataset, query, choices)
    return key


class CombiningRule:
    def __init__(self):
        self.evaluation_meta_data_file = \
            '/scratch/c.scmnk4/elexir/resources/' \
            'sim_importance_datasets_second_appr_composition_40960_glove_numberbatch_top5.pkl'
        # self.evaluation_meta_data_file = \
        #     '/scratch/c.scmnk4/elexir/resources/' \
        #     'sim_importance_datasets_second_appr_sum_max_min_glove_numberbatch_top5.pkl'
        # self.evaluation_meta_data_file = \
        #     '/scratch/c.scmnk4/elexir/resources/' \
        #     'sim_importance_datasets_second_appr_transformer_2048_glove_numberbatch_top5.pkl'
        self.relbert_embeddings_for_concept_pairs = dict()
        self.composition_model_embeddings_for_concept_pairs = dict()
        self.concepts_in_path_length_2 = dict()
        self.classifier = Classifier()
        self.classifier.load_model()

    def load_meta_data(self, relbert_emb, composition_emb, related_concepts):
        for key in relbert_emb:
            self.relbert_embeddings_for_concept_pairs[key] = relbert_emb[key]
        for key in composition_emb:
            self.composition_model_embeddings_for_concept_pairs[key] = composition_emb[key]
        for key in related_concepts:
            self.concepts_in_path_length_2[key] = related_concepts[key]

    def evaluate_performance_on_partitioned_dataset(self):
        appr1_p1 = []
        appr2_p1 = []
        appr1_p2 = []
        appr2_p2 = []
        appr1_p3 = []
        appr2_p3 = []
        appr1_p4 = []
        appr2_p4 = []

        appr1_p5 = []
        appr2_p5 = []

        appr1_p6 = []
        appr2_p6 = []

        appr1_p7 = []
        appr2_p7 = []

        appr1_p8 = []
        appr2_p8 = []

        appr1_p9 = []
        appr2_p9 = []

        appr1_p10 = []
        appr2_p10 = []

        with open(self.evaluation_meta_data_file, 'rb') as f:
            dataset_results = pickle.load(f)
        with open('best_choice_importance_lt_0.1.txt', 'w') as file1, \
                open('best_choice_importance_lt_0.2.txt', 'w') as file2, \
                open('best_choice_importance_lt_0.3.txt', 'w') as file3, \
                open('best_choice_importance_lt_0.4.txt', 'w') as file4, \
                open('best_choice_importance_lt_0.5.txt', 'w') as file5, \
                open('best_choice_importance_lt_0.6.txt', 'w') as file6, \
                open('best_choice_importance_lt_0.7.txt', 'w') as file7, \
                open('best_choice_importance_lt_0.8.txt', 'w') as file8, \
                open('best_choice_importance_lt_0.9.txt', 'w') as file9, \
                open('best_choice_importance_lt_1.0.txt', 'w') as file10:
            for dataset in dataset_results:
                if 'dataset' not in dataset:
                    continue
                if dataset['dataset'] != 'sat' and dataset['dataset'] != 'u2' and dataset['dataset'] != 'u4' \
                        and dataset['dataset'] != 'bats' and dataset['dataset'] != 'google' and \
                        dataset['dataset'] != 'scan' and dataset['dataset'] != 'ekar':
                    continue
                # if dataset['dataset'] != 'sat':
                #     continue
                dataset_name = dataset['dataset']
                records = dataset['records']
                combined_approach = []
                approach_1_accuracy = []
                approach_2_accuracy = []
                for record in records:
                    approach2_prediction = record[-3]
                    answer = record[-2]
                    stem_importance = record[0]
                    cosine_importance_of_choices1 = record[1:-3]
                    cosine_importance_of_choices = [cosine_importance_of_choices1[i:i + 2]
                                                    for i in range(0, len(cosine_importance_of_choices1), 2)]
                    choice_cosine = -2
                    choice_importance = -1
                    cosines = []
                    for [cosine, importance] in cosine_importance_of_choices:
                        if cosine > choice_cosine:
                            choice_cosine = cosine
                            choice_importance = importance
                        cosines.append(cosine)
                    approach1_prediction = cosines.index(choice_cosine)
                    item = f'Dataset = {dataset_name} \t Best choice according to RelBERT = {approach1_prediction} ' \
                           f'\t Approach 2 Prediction = {approach2_prediction} \t Analogy Question = {record[-1]}\n'
                    # if (stem_importance > 0.75 and choice_importance < 0.5) or (choice_importance < 0.1):
                    if abs(stem_importance - choice_importance) > 0.25:
                        # if choice_importance < 0.5:
                        approach_1_accuracy.append(approach1_prediction == answer)
                        approach_2_accuracy.append(approach2_prediction == answer)
                        combined_approach.append(approach2_prediction == answer)
                    else:
                        approach_1_accuracy.append(approach1_prediction == answer)
                        approach_2_accuracy.append(approach2_prediction == answer)
                        combined_approach.append(approach1_prediction == answer)

                    if choice_importance < 0.1:
                        appr1_p1.append(approach1_prediction == answer)
                        appr2_p1.append(approach2_prediction == answer)
                        file1.write(item)
                    elif choice_importance < 0.2:
                        appr1_p2.append(approach1_prediction == answer)
                        appr2_p2.append(approach2_prediction == answer)
                        file2.write(item)
                    elif choice_importance < 0.3:
                        appr1_p3.append(approach1_prediction == answer)
                        appr2_p3.append(approach2_prediction == answer)
                        file3.write(item)
                    elif choice_importance < 0.4:
                        appr1_p4.append(approach1_prediction == answer)
                        appr2_p4.append(approach2_prediction == answer)
                        file4.write(item)
                    elif choice_importance < 0.5:
                        appr1_p5.append(approach1_prediction == answer)
                        appr2_p5.append(approach2_prediction == answer)
                        file5.write(item)
                    elif choice_importance < 0.6:
                        appr1_p6.append(approach1_prediction == answer)
                        appr2_p6.append(approach2_prediction == answer)
                        file6.write(item)
                    elif choice_importance < 0.7:
                        appr1_p7.append(approach1_prediction == answer)
                        appr2_p7.append(approach2_prediction == answer)
                        file7.write(item)
                    elif choice_importance < 0.8:
                        appr1_p8.append(approach1_prediction == answer)
                        appr2_p8.append(approach2_prediction == answer)
                        file8.write(item)
                    elif choice_importance < 0.9:
                        appr1_p9.append(approach1_prediction == answer)
                        appr2_p9.append(approach2_prediction == answer)
                        file9.write(item)
                    else:
                        appr1_p10.append(approach1_prediction == answer)
                        appr2_p10.append(approach2_prediction == answer)
                        file10.write(item)
                print('Dataset = %s, Approach 1 Accuracy = %s', dataset_name,
                      str(calculate_accuracy(approach_1_accuracy)))
                print('Dataset = %s, Approach 2 Accuracy = %s', dataset_name,
                      str(calculate_accuracy(approach_2_accuracy)))
                print('Dataset = %s, Combined Approach Accuracy = %s', dataset_name,
                      str(calculate_accuracy(combined_approach)))
        print('Approach 1 Percentile 1 = %s', str(calculate_accuracy(appr1_p1)))
        print('Approach 2 Percentile 1 = %s', str(calculate_accuracy(appr2_p1)))
        print('Approach 1 Percentile 2 = %s', str(calculate_accuracy(appr1_p2)))
        print('Approach 2 Percentile 2 = %s', str(calculate_accuracy(appr2_p2)))
        print('Approach 1 Percentile 3 = %s', str(calculate_accuracy(appr1_p3)))
        print('Approach 2 Percentile 3 = %s', str(calculate_accuracy(appr2_p3)))
        print('Approach 1 Percentile 4 = %s', str(calculate_accuracy(appr1_p4)))
        print('Approach 2 Percentile 4 = %s', str(calculate_accuracy(appr2_p4)))
        print('Approach 1 Percentile 5 = %s', str(calculate_accuracy(appr1_p5)))
        print('Approach 2 Percentile 5 = %s', str(calculate_accuracy(appr2_p5)))

        print('Approach 1 Percentile 6 = %s', str(calculate_accuracy(appr1_p6)))
        print('Approach 2 Percentile 6 = %s', str(calculate_accuracy(appr2_p6)))

        print('Approach 1 Percentile 7 = %s', str(calculate_accuracy(appr1_p7)))
        print('Approach 2 Percentile 7 = %s', str(calculate_accuracy(appr2_p7)))

        print('Approach 1 Percentile 8 = %s', str(calculate_accuracy(appr1_p8)))
        print('Approach 2 Percentile 8 = %s', str(calculate_accuracy(appr2_p8)))

        print('Approach 1 Percentile 9 = %s', str(calculate_accuracy(appr1_p9)))
        print('Approach 2 Percentile 9 = %s', str(calculate_accuracy(appr2_p9)))

        print('Approach 1 Percentile 10 = %s', str(calculate_accuracy(appr1_p10)))
        print('Approach 2 Percentile 10 = %s', str(calculate_accuracy(appr2_p10)))

    def combining_rule(self, relbert_embd, composition_emb):
        importance = self.classifier.importance(relbert_embd)
        k0 = 100.0
        i0 = 0.75
        score = 1 / (1 + np.exp(-1 * k0 * (importance - i0)))
        score = importance ** 2
        combined_embd = score * normalize(np.array(relbert_embd).reshape(1, -1)) + (1 - score) * normalize(
            np.array(composition_emb).reshape(1, -1))
        combined_embd = combined_embd[0].tolist()
        return combined_embd

    def combining_rule_prediction(self, stem, choice):
        [a, b] = stem
        related_ab = self.concepts_in_path_length_2[(a, b)]
        triangle = False
        for [c, d] in choice:
            related_cd = self.concepts_in_path_length_2[(c, d)]
            if len(related_cd) > 0:
                triangle = True
                break
        if len(related_ab) > 0 and triangle:
            predictions = []
            ab_predicted_embd = self.combining_rule(self.relbert_embeddings_for_concept_pairs[(a, b)],
                                                    self.composition_model_embeddings_for_concept_pairs[(a, b)])
            for [c, d] in choice:
                related_cd = self.concepts_in_path_length_2[(c, d)]
                if len(related_cd) > 0:
                    cd_predicted_embd = \
                        self.combining_rule(self.relbert_embeddings_for_concept_pairs[(c, d)],
                                            self.composition_model_embeddings_for_concept_pairs[(c, d)])
                    tempsim = min(cosine_similarity(ab_predicted_embd, cd_predicted_embd),
                                  cosine_similarity(self.relbert_embeddings_for_concept_pairs[(a, b)],
                                                    self.relbert_embeddings_for_concept_pairs[(c, d)]))
                else:
                    tempsim = -1
                predictions.append(tempsim)
            prediction = predictions.index(max(predictions))
        else:
            sim = [cosine_similarity(self.relbert_embeddings_for_concept_pairs[(a, b)],
                                     self.relbert_embeddings_for_concept_pairs[(c, d)]) for [c, d] in choice]
            prediction = sim.index(max(sim))
        return prediction

    def combining_rule_prediction2(self, stem, choice):
        return np.random.permutation(len(choice))[0]
        imp = []
        for [c, d] in choice:
            imp.append(self.classifier.importance(self.relbert_embeddings_for_concept_pairs[(c, d)]))
        return imp.index(max(imp))

    def estimate_combining_rule_for_relbert_and_composition(self):
        with open(self.evaluation_meta_data_file, 'rb') as f:
            meta_data = pickle.load(f)
        relbert_emb = None
        composition_emb = None
        related_concepts = None
        # relbert_emb = meta_data[-3]
        # composition_emb = meta_data[-2]
        # related_concepts = meta_data[-1]
        for data in meta_data:
            if 'relbert' in data:
                relbert_emb = data['relbert']
            if 'composition' in data:
                composition_emb = data['composition']
            if 'related' in data:
                related_concepts = data['related']
        self.load_meta_data(relbert_emb, composition_emb, related_concepts)
        for dataset in meta_data:
            if 'dataset' in dataset:
                result = []
                dataset_name = dataset['dataset']
                records = dataset['records']

                for record in records:
                    item = record[-1]
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
                    combining_rule_answer = self.combining_rule_prediction(stem, choice)
                    answer = item['answer']
                    result.append(combining_rule_answer == answer)
                [size, accuracy] = calculate_accuracy(result)
                print(f'Dataset = {dataset_name}, Total number of questions = {size}, Accuracy = {accuracy}')

    def read_gpt_4_results(self):
        gpt_4_prediction_files = ["/scratch/c.scmnk4/elexir/resources/results/gpt3_correct_predictions.txt",
                                  "/scratch/c.scmnk4/elexir/resources/results/gpt3_incorrect_predictions.txt"]
        self.gpt_4_results = {}
        for filename in gpt_4_prediction_files:
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
                        self.gpt_4_results[key] = int(predicted_answer)

    def get_gpt4_results(self, dataset_name, stem, choice):
        key = get_lookup_key(dataset_name, stem, choice)
        return self.gpt_4_results[key]

    def analyze_gpt_4_results(self):
        self.read_gpt_4_results()
        with open(self.evaluation_meta_data_file, 'rb') as f:
            meta_data = pickle.load(f)
        relbert_emb = None
        composition_emb = None
        related_concepts = None
        for data in meta_data:
            if 'relbert' in data:
                relbert_emb = data['relbert']
            if 'composition' in data:
                composition_emb = data['composition']
            if 'related' in data:
                related_concepts = data['related']
        self.load_meta_data(relbert_emb, composition_emb, related_concepts)
        for dataset in meta_data:
            if 'dataset' in dataset:
                result = []
                if dataset['dataset'] != 'sat' and dataset['dataset'] != 'u2' and dataset['dataset'] != 'u4' \
                        and dataset['dataset'] != 'bats' and dataset['dataset'] != 'google':
                    continue
                dataset_name = dataset['dataset']
                records = dataset['records']
                for record in records:
                    item = record[-1]
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
                    combining_rule_answer = self.get_gpt4_results(dataset_name, stem, choice)
                    answer = item['answer']
                    result.append(combining_rule_answer == answer)
                [size, accuracy] = calculate_accuracy(result)
                print(f'Dataset = {dataset_name}, Total number of questions = {size}, Accuracy = {accuracy}')


if __name__ == '__main__':
    obj = CombiningRule()
    # obj.estimate_combining_rule_for_relbert_and_composition()
    obj.evaluate_performance_on_partitioned_dataset()
    # obj.analyze_gpt_4_results()
