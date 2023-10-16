import pickle
import unidecode
from relbert import cosine_similarity

from reasoning_with_vectors.core.analysis import calculate_accuracy
from reasoning_with_vectors.experiments.evaluation import get_lookup_key
from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.importance.importance_filter import Classifier
from reasoning_with_vectors.core.utils import read_datasets


class Percentile:
    def __init__(self, results_file):
        self.results_file = results_file
        self.relbert_embeddings_for_concept_pairs = dict()
        self.composition_model_embeddings_for_concept_pairs = dict()
        self.concepts_in_path_length_2 = dict()
        self.results = dict()
        self.classifier = Classifier()
        self.classifier.load_model()

    def load_meta_data(self, relbert_emb, composition_emb, related_concepts, results):
        for key in relbert_emb:
            self.relbert_embeddings_for_concept_pairs[key] = relbert_emb[key]
        for key in composition_emb:
            self.composition_model_embeddings_for_concept_pairs[key] = composition_emb[key]
        for key in related_concepts:
            self.concepts_in_path_length_2[key] = related_concepts[key]
        for key in results:
            self.results[key] = results[key]

    def analyze_relbert(self, stem, choice):
        [a, b] = stem
        stem_imp = self.classifier.importance(self.relbert_embeddings_for_concept_pairs[(a, b)])
        sim = [cosine_similarity(self.relbert_embeddings_for_concept_pairs[(a, b)],
                                 self.relbert_embeddings_for_concept_pairs[(c, d)]) for [c, d] in choice]
        prediction = sim.index(max(sim))
        [c, d] = choice[prediction]
        best_choice_imp = self.classifier.importance(self.relbert_embeddings_for_concept_pairs[(c, d)])
        return [stem_imp, prediction, best_choice_imp]

    def evaluate_performance_on_partitioned_dataset(self):
        r_partition1 = []
        p_partition1 = []
        r_partition2 = []
        p_partition2 = []
        r_partition3 = []
        p_partition3 = []
        r_partition4 = []
        p_partition4 = []
        r_partition5 = []
        p_partition5 = []
        r_partition6 = []
        p_partition6 = []
        r_partition7 = []
        p_partition7 = []
        r_partition8 = []
        p_partition8 = []
        r_partition9 = []
        p_partition9 = []
        r_partition10 = []
        p_partition10 = []
        with open(self.results_file, 'rb') as f:
            meta_data = pickle.load(f)
        relbert_emb = None
        composition_emb = None
        related_concepts = None
        results = None
        for data in meta_data:
            if 'relbert' in data:
                relbert_emb = data['relbert']
            if 'composition' in data:
                composition_emb = data['composition']
            if 'related' in data:
                related_concepts = data['related']
            if 'predictions' in data:
                results = data['predictions']
        self.load_meta_data(relbert_emb, composition_emb, related_concepts, results)
        with open('results/partition1.txt', 'w') as file1, \
                open('results/partition2.txt', 'w') as file2, \
                open('results/partition3.txt', 'w') as file3, \
                open('results/partition4.txt', 'w') as file4, \
                open('results/partition5.txt', 'w') as file5, \
                open('results/partition6.txt', 'w') as file6, \
                open('results/partition7.txt', 'w') as file7, \
                open('results/partition8.txt', 'w') as file8, \
                open('results/partition9.txt', 'w') as file9, \
                open('results/partition10.txt', 'w') as file10:
            for dataset_name in configuration.analogy_datasets:
                all_choices = []
                data = read_datasets(dataset_name, approach='only_test')
                # data = read_datasets(dataset_name, approach='test_and_valid')
                r_overall_accuracy = []
                p_overall_accuracy = []
                o_overall_accuracy = []

                # r_partition1 = []
                # p_partition1 = []
                # r_partition2 = []
                # p_partition2 = []
                # r_partition3 = []
                # p_partition3 = []
                # r_partition4 = []
                # p_partition4 = []
                # r_partition5 = []
                # p_partition5 = []
                # r_partition6 = []
                # p_partition6 = []
                # r_partition7 = []
                # p_partition7 = []
                # r_partition8 = []
                # p_partition8 = []
                # r_partition9 = []
                # p_partition9 = []
                # r_partition10 = []
                # p_partition10 = []

                for item in data:
                    answer = item['answer']
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
                    all_choices = all_choices + choice

                    [stem_importance, r_predicted_answer, best_choice_importance] = self.analyze_relbert(stem, choice)

                    key = get_lookup_key(dataset_name, stem, choice)
                    p_predicted_answer = self.results[key]

                    item = f'Dataset = {dataset_name} \t Best choice according to RelBERT = {r_predicted_answer} ' \
                           f'\t Analogy Question = {item}\n'

                    if min(best_choice_importance, stem_importance) < 0.5:
                        r_overall_accuracy.append(r_predicted_answer == answer)
                        p_overall_accuracy.append(p_predicted_answer == answer)
                        o_overall_accuracy.append(p_predicted_answer == answer)
                    else:
                        r_overall_accuracy.append(r_predicted_answer == answer)
                        p_overall_accuracy.append(p_predicted_answer == answer)
                        o_overall_accuracy.append(r_predicted_answer == answer)

                    if min(best_choice_importance, stem_importance) < 0.25:
                        r_partition1.append(r_predicted_answer == answer)
                        p_partition1.append(p_predicted_answer == answer)
                        file1.write(item)
                    elif min(best_choice_importance, stem_importance) < 0.5:
                        r_partition2.append(r_predicted_answer == answer)
                        p_partition2.append(p_predicted_answer == answer)
                        file2.write(item)
                    elif min(best_choice_importance, stem_importance) < 0.75:
                        r_partition3.append(r_predicted_answer == answer)
                        p_partition3.append(p_predicted_answer == answer)
                        file3.write(item)
                    else:
                        r_partition4.append(r_predicted_answer == answer)
                        p_partition4.append(p_predicted_answer == answer)
                        file4.write(item)

                    # if min(best_choice_importance, stem_importance) < 0.1:
                    #     r_partition1.append(r_predicted_answer == answer)
                    #     p_partition1.append(p_predicted_answer == answer)
                    #     file1.write(item)
                    # elif min(best_choice_importance, stem_importance) < 0.2:
                    #     r_partition2.append(r_predicted_answer == answer)
                    #     p_partition2.append(p_predicted_answer == answer)
                    #     file2.write(item)
                    # elif min(best_choice_importance, stem_importance) < 0.3:
                    #     r_partition3.append(r_predicted_answer == answer)
                    #     p_partition3.append(p_predicted_answer == answer)
                    #     file3.write(item)
                    # elif min(best_choice_importance, stem_importance) < 0.4:
                    #     r_partition4.append(r_predicted_answer == answer)
                    #     p_partition4.append(p_predicted_answer == answer)
                    #     file4.write(item)
                    # elif min(best_choice_importance, stem_importance) < 0.5:
                    #     r_partition5.append(r_predicted_answer == answer)
                    #     p_partition5.append(p_predicted_answer == answer)
                    #     file5.write(item)
                    # elif min(best_choice_importance, stem_importance) < 0.6:
                    #     r_partition6.append(r_predicted_answer == answer)
                    #     p_partition6.append(p_predicted_answer == answer)
                    #     file6.write(item)
                    # elif min(best_choice_importance, stem_importance) < 0.7:
                    #     r_partition7.append(r_predicted_answer == answer)
                    #     p_partition7.append(p_predicted_answer == answer)
                    #     file7.write(item)
                    # elif min(best_choice_importance, stem_importance) < 0.8:
                    #     r_partition8.append(r_predicted_answer == answer)
                    #     p_partition8.append(p_predicted_answer == answer)
                    #     file8.write(item)
                    # elif min(best_choice_importance, stem_importance) < 0.9:
                    #     r_partition9.append(r_predicted_answer == answer)
                    #     p_partition9.append(p_predicted_answer == answer)
                    #     file9.write(item)
                    # else:
                    #     r_partition10.append(r_predicted_answer == answer)
                    #     p_partition10.append(p_predicted_answer == answer)
                    #     file10.write(item)

                print(f'Dataset = {dataset_name}, RelBERT Accuracy = {calculate_accuracy(r_overall_accuracy)}')

                print(f'Dataset = {dataset_name}, Approach Accuracy = {calculate_accuracy(p_overall_accuracy)}')

                print(f'Dataset = {dataset_name}, Combined Accuracy = {calculate_accuracy(o_overall_accuracy)}')
                print(f'Dataset = {dataset_name}, Average number of choices = {len(all_choices)/len(data)}')
                print(f'Dataset = {dataset_name}, Number of questions = {len(data)}')
                print('-------------------------------------------------------------')
                print('-------------------------------------------------------------')

                print(f'RelBERT Percentile 1 = {calculate_accuracy(r_partition1)}')
                print(f'Approach Percentile 1 = {calculate_accuracy(p_partition1)}')
                print('\n')
                print(f'RelBERT Percentile 2 = {calculate_accuracy(r_partition2)}')
                print(f'Approach Percentile 2 = {calculate_accuracy(p_partition2)}')
                print('\n')
                print(f'RelBERT Percentile 3 = {calculate_accuracy(r_partition3)}')
                print(f'Approach Percentile 3 = {calculate_accuracy(p_partition3)}')
                print('\n')
                print(f'RelBERT Percentile 4 = {calculate_accuracy(r_partition4)}')
                print(f'Approach Percentile 4 = {calculate_accuracy(p_partition4)}')
                # print('\n')
                # print(f'RelBERT Percentile 5 = {calculate_accuracy(r_partition5)}')
                # print(f'Approach Percentile 5 = {calculate_accuracy(p_partition5)}')
                # print('\n')
                # print(f'RelBERT Percentile 6 = {calculate_accuracy(r_partition6)}')
                # print(f'Approach Percentile 6 = {calculate_accuracy(p_partition6)}')
                # print('\n')
                # print(f'RelBERT Percentile 7 = {calculate_accuracy(r_partition7)}')
                # print(f'Approach Percentile 7 = {calculate_accuracy(p_partition7)}')
                # print('\n')
                # print(f'RelBERT Percentile 8 = {calculate_accuracy(r_partition8)}')
                # print(f'Approach Percentile 8 = {calculate_accuracy(p_partition8)}')
                # print('\n')
                # print(f'RelBERT Percentile 9 = {calculate_accuracy(r_partition9)}')
                # print(f'Approach Percentile 9 = {calculate_accuracy(p_partition9)}')
                # print('\n')
                # print(f'RelBERT Percentile 10 = {calculate_accuracy(r_partition10)}')
                # print(f'Approach Percentile 10 = {calculate_accuracy(p_partition10)}')
                print('-------------------------------------------------------------\n')


if __name__ == '__main__':
    the_results_file = '/scratch/c.scmnk4/elexir/resources/results/dp4_Sum_Max_Min.pkl'
    obj = Percentile(the_results_file)
    obj.evaluate_performance_on_partitioned_dataset()
