import os
import json
from reasoning_with_vectors.conf import configuration


def read_datasets(dataset1, approach='only_test'):
    dataset_dir = os.path.join(configuration.analogy_datasets_path, dataset1)
    dataset_test = os.path.join(dataset_dir, 'test.jsonl')
    dataset_valid = os.path.join(dataset_dir, 'valid.jsonl')
    f = open(dataset_test, 'r')
    data = f.read()
    f.close()
    data = data.split('\n')
    if approach == 'only_valid':
        f = open(dataset_valid, 'r')
        data = f.read()
        f.close()
        data = data.split('\n')
    if approach == 'only_test':
        f = open(dataset_test, 'r')
        data = f.read()
        f.close()
        data = data.split('\n')
    elif approach == 'test_and_valid':
        f = open(dataset_test, 'r')
        data = f.read()
        f.close()
        data = data.split('\n')
        f = open(dataset_valid, 'r')
        data1 = f.read()
        f.close()
        data1 = data1.split('\n')
        data = data + data1
    data_new = []
    for item in data:
        if item == '':
            pass
        else:
            itemf = json.loads(item)
            data_new.append(itemf)
    return data_new


def merge_files():
    with open('/scratch/c.scmnk4/elexir/resources/results/gpt-4_correct_predictions.txt', 'w') as outfile:
        for fname in \
                ['/scratch/c.scmnk4/elexir/resources/results/gpt3_correct_predictions_sat_u2_u4_bats_google_scan.txt',
                 '/scratch/c.scmnk4/elexir/resources/results/gpt-4-turbo_correct_predictions_ekar_only.txt']:
            with open(fname, 'r') as infile:
                for line in infile:
                    outfile.write(line)

    with open('/scratch/c.scmnk4/elexir/resources/results/gpt-4_incorrect_predictions.txt', 'w') as outfile:
        for fname in \
                ['/scratch/c.scmnk4/elexir/resources/results/gpt3_incorrect_predictions_sat_u2_u4_bats_google_scan.txt',
                 '/scratch/c.scmnk4/elexir/resources/results/gpt-4-turbo_incorrect_predictions_ekar_only.txt']:
            with open(fname, 'r') as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == '__main__':
    merge_files()
