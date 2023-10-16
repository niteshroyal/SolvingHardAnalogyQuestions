import random

from reasoning_with_vectors.core.data_processor import TrainingDataProcessor
from reasoning_with_vectors.importance.importance_filter import Classifier


class Visualization(TrainingDataProcessor):
    def __init__(self):
        super().__init__()
        self.classifier = Classifier()
        self.classifier.load_model()

    def get_importance(self, embedding):
        return self.classifier.importance(embedding)

    def is_conceptnet_link(self, c1, c2):
        answer = False
        qry = f'conceptnet_edge("{c1}", "{c2}").'
        for sol in self.prolog.query(qry):
            answer = True
            break
        return answer

    def get_all_conceptnet_links(self):
        conceptnet_links = set()
        qry = 'conceptnet_edge(X,Y).'
        for sol in self.prolog.query(qry):
            c1 = sol['X'].decode('UTF-8')
            c2 = sol['Y'].decode('UTF-8')
            conceptnet_links.add((c1, c2))
            if len(conceptnet_links) > 10000:
                break
        class1 = []
        class2 = []
        class3 = []
        counter = 0
        for (a, b) in conceptnet_links:
            counter += 1
            imp = self.get_importance(self.get_embedding_for_one_pair([a, b]))
            if imp > 0.90:
                class1.append([a, b, imp])
            elif 0.45 < imp < 0.55:
                class2.append([a, b, imp])
            elif imp < 0.1:
                class3.append([a, b, imp])
            else:
                pass
            if len(class1) >= 10 and len(class2) >= 10 and len(class3) >= 10:
                break
            if counter % 100 == 0:
                print(f'Counter = {counter}, Class1 size = {len(class1)}, '
                      f'Class2 size = {len(class2)}, Class3 size = {len(class3)}')
        print(f'Class1 = {class1}')
        print(f'Class2 = {class2}')
        print(f'Class3 = {class3}')

    def get_some_conceptnet_edges(self, num=10):
        class1 = []
        class2 = []
        class3 = []
        counter = 0
        for concept_pair, embedding in self.iter_stored_relbert_embds():
            counter += 1
            [c1, c2] = concept_pair
            if self.is_conceptnet_link(c1, c2) and 3 < len(c1) < 12 and 3 < len(c2) < 12:
                imp = self.get_importance(embedding)
                if 0.985 < imp < 0.995:
                    class1.append([c1, c2])
                elif 0.495 < imp < 0.505:
                    class2.append([c1, c2])
                elif 0.005 < imp < 0.015:
                    class3.append([c1, c2])
                else:
                    pass
            if len(class1) >= num and len(class2) >= num and len(class3) >= num:
                break
            if counter % 1000 == 0:
                print(f'Counter = {counter}, Class1 size = {len(class1)}, '
                      f'Class2 size = {len(class2)}, Class3 size = {len(class3)}')
        print(f'Class1 = {class1}')
        print(f'Class2 = {class2}')
        print(f'Class3 = {class3}')


if __name__ == '__main__':
    v_obj = Visualization()
    v_obj.get_all_conceptnet_links()
    # v_obj.get_some_conceptnet_edges()
