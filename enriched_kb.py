import os
import re
import logging
import json
from pyswip import Prolog

from pythonProject.research.reasoning_with_vectors.conf import configuration

relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]


def load_merge_relation():
    relation_mapping = dict()
    for line in relation_groups:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping


def dataset_type():
    new_data = {
        "dataset": "",
        "license": "",
        "sources": [
            {
                "contributor": "",
                "process": ""
            }
        ],
        "weight": configuration.importance_filter_threshold
    }
    return json.dumps(new_data) + '\n'


default_relation_type = '/r/c-1'


def relation_type(cluster):
    return f'/r/{cluster}'


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def get_concepts_qagnn_experiment():
    qa_concept_list = []
    with open(configuration.qa_vocab_file, "r") as file:
        for line in file:
            concept = line.strip()
            qa_concept_list.append(concept)
    return qa_concept_list


def replace_relation(input_text, new_relation):
    new_relation = new_relation + '/'
    return re.sub(r'/r/[^/]+/', new_relation, input_text)


def english(txt):
    try:
        if txt[0:6] == '/c/en/':
            return True
        else:
            return False
    except:
        return False


def scrap_english_text(txt):
    txt = txt.split('/')
    txt = txt[3]
    txt = txt.replace('_', ' ')
    return txt


class EnrichedKB:
    def __init__(self):
        self.new_edges = None
        self.prolog = None
        self.dataset_type = dataset_type()
        self.qa_concepts = get_concepts_qagnn_experiment()
        self.old_conceptnet_edges = set()
        self.relation_mapping = load_merge_relation()

    def start_prolog_client(self):
        self.prolog = Prolog()
        self.prolog.consult(configuration.enriched_kb_with_clusters)

    def enriched_lines_with_clusters(self, line):
        line = line.strip().split('\t')
        rel = line[1].split("/")[-1].lower()
        if rel not in self.relation_mapping:
            pass
        else:
            if english(line[2]) and english(line[3]):
                c1 = scrap_english_text(line[2])
                c2 = scrap_english_text(line[3])
                qry = f't("{c1}","{c2}",Z).'
                cluster = None
                for sol in self.prolog.query(qry):
                    cluster = sol['Z'].decode('UTF-8')
                    break
                if cluster is not None:
                    line[1] = relation_type(cluster)
                    line[0] = replace_relation(line[0], line[1])
                    self.old_conceptnet_edges.add((c1, c2))
                else:
                    line[1] = default_relation_type
                    line[0] = replace_relation(line[0], line[1])
            else:
                line[1] = default_relation_type
                line[0] = replace_relation(line[0], line[1])
        line[4] = dataset_type()
        return '\t'.join(line)

    def enriched_conceptnet(self):
        self.start_prolog_client()
        self.extract_new_edges()
        with open(configuration.original_conceptnet_used_in_qagnn, 'r') as f_orig, \
                open(configuration.enriched_conceptnet_with_clusters, 'w', newline='') as f_new:
            for line in f_orig:
                line = self.enriched_lines_with_clusters(line)
                f_new.write(line)
            for edge in self.new_edges:
                e = (scrap_english_text(edge[2]), scrap_english_text(edge[3]))
                if e in self.old_conceptnet_edges:
                    continue
                f_new.write('\t'.join(edge))

    def format_new_edge(self, c1, c2, label):
        r = '/r/' + str(label)
        c1 = c1.replace(" ", "_")
        c2 = c2.replace(" ", "_")
        if c1 not in self.qa_concepts or c2 not in self.qa_concepts:
            return None
        else:
            pass
        c1 = f'/c/en/{c1}'
        c2 = f'/c/en/{c2}'
        d = f'/a/[{r}/,{c1}/,{c2}/]'
        return [d, r, c1, c2, self.dataset_type]

    def extract_new_edges(self):
        unique_clusters = set()
        self.new_edges = []
        qry = 't(X,Y,Z).'
        counter = 0
        for sol in self.prolog.query(qry):
            c1 = sol['X'].decode('UTF-8')
            c2 = sol['Y'].decode('UTF-8')
            label = sol['Z'].decode('UTF-8')
            unique_clusters.add(label)
            new_edge = self.format_new_edge(c1, c2, label)
            if new_edge is None:
                pass
            else:
                self.new_edges.append(new_edge)
            counter += 1
            if counter % 10000 == 0:
                logging.info('Processed edges = %d', counter)
        logging.info(f'The number of unique relations in '
                     f'{configuration.enriched_conceptnet_with_clusters} is {len(unique_clusters)}')


if __name__ == '__main__':
    initialization()
    obj = EnrichedKB()
    obj.enriched_conceptnet()
