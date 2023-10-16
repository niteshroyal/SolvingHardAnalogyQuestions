import logging
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reasoning_with_vectors.conf import configuration

if configuration.composition_model == 'ComplexCompositionModel':
    from reasoning_with_vectors.condensed.complex_composition import ComplexCompositionModel as Model
elif configuration.composition_model == 'CompositionModelInverse':
    from reasoning_with_vectors.condensed.composition_with_inverse import CompositionModelInverse as Model
elif configuration.composition_model == 'CompositionModel':
    from reasoning_with_vectors.condensed.composition import CompositionModel as Model
elif configuration.composition_model == 'SetTransformer':
    from reasoning_with_vectors.condensed.set_transformer import SetTransformer as Model
else:
    raise Exception('Problem with configuration.composition_model')


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            f"{os.path.splitext(os.path.basename(__file__))[0]}_"
                            f"DPApr{configuration.dataset_preparation_approch_for_composition_model}.log")
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=log_file, filemode='w', level=logging.INFO)
    with open(configuration.configuration_file_to_consider, 'r') as f:
        conf = f.read()
    logging.info(conf)


def shuffle_dataset(dataset):
    shuffled_index = np.random.permutation(len(dataset))
    shuffled_dataset = [dataset[i] for i in shuffled_index]
    return shuffled_dataset


def shuffle_files(files):
    np.random.seed(42)
    shuffled_index = np.random.permutation(len(files))
    shuffled_files = [files[i] for i in shuffled_index]
    return shuffled_files


def get_training_file():
    dir_name = os.path.dirname(configuration.training_dataset)
    filename_without_ext = os.path.splitext(os.path.basename(configuration.training_dataset))[0]
    files_temp = [filename for filename in os.listdir(dir_name) if filename.startswith(filename_without_ext)]
    files_temp.sort()
    files = []
    for afile in files_temp:
        files.append(os.path.join(dir_name, afile))
    return files


class PathLength2Training(Model):
    def __init__(self, run_id=0):
        self.run_id = run_id
        super().__init__()
        self.overfit = False

    def init_model(self):
        self.model_file = os.path.join(configuration.model_save_path,
                                       f'{Model.__name__}_'
                                       f'DPApr{configuration.dataset_preparation_approch_for_composition_model}_'
                                       f'Run{self.run_id}_'
                                       f'Dim{configuration.inner_layer_dimension}.pth')
        self.loss = nn.CosineSimilarity(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=configuration.learning_rate)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_gpus = torch.cuda.device_count()
        if self.device == 'cuda':
            logging.info(f"#Run{self.run_id} There are {num_gpus} GPUs:")
            for gpu_number in range(num_gpus):
                logging.info(f"#Run{self.run_id}   GPU {gpu_number}: {torch.cuda.get_device_name(gpu_number)}")
            logging.info(f"#Run{self.run_id} {Model.__name__} currently can not utilize mutiple GPUs. "
                         f"So, it is running on one GPU\n")
        else:
            logging.info(f"#Run{self.run_id} {Model.__name__} is running on CPU\n")
        self.to(self.device)
        if os.path.exists(self.model_file):
            self.load_model()
            logging.info(f"#Run{self.run_id} Loaded model weights from {self.model_file}")
        else:
            logging.info(f"#Run{self.run_id} Model file does not exist. Initializing weights randomly.")

    def run_session(self, dataset, session='training'):
        losses = []
        xz_list1 = []
        zy_list1 = []
        segmentid_list = []
        xy_list = []
        segmentid = -1
        for [_, _, xy, xzy] in dataset:
            segmentid += 1
            xy_list.append(xy)
            for [_, xz1, zy1] in xzy:
                xz_list1.append(xz1)
                zy_list1.append(zy1)
                segmentid_list.append(segmentid)
            if len(segmentid_list) >= configuration.training_batch_size:
                if session == 'training':
                    if self.overfit:
                        losses.append(self.run_training(xz_list1, zy_list1, segmentid_list, xy_list, 10))
                    else:
                        losses.append(self.run_training(xz_list1, zy_list1, segmentid_list, xy_list, 1))
                else:
                    losses.append(self.run_validation(xz_list1, zy_list1, segmentid_list, xy_list))
                xz_list1 = []
                zy_list1 = []
                segmentid_list = []
                xy_list = []
                segmentid = -1
        if len(segmentid_list) > 0:
            if session == 'training':
                if self.overfit:
                    losses.append(self.run_training(xz_list1, zy_list1, segmentid_list, xy_list, 10))
                else:
                    losses.append(self.run_training(xz_list1, zy_list1, segmentid_list, xy_list, 1))
            else:
                losses.append(self.run_validation(xz_list1, zy_list1, segmentid_list, xy_list))
        return np.mean(losses)

    def train_model(self):
        files = get_training_file()
        files = shuffle_files(files)
        num_validation_files = int(len(files) * 0.1)
        training_files = files[num_validation_files:]
        validation_files = files[0:num_validation_files]
        logging.info(f"#Run{self.run_id} Number of training data file parts = {len(training_files)}")
        for epoch in range(1, configuration.number_of_epochs + 1):
            training_files = shuffle_files(training_files)
            training_losses = []
            for afile in training_files:
                with open(afile, 'rb') as handle:
                    dataset = pickle.load(handle)
                dataset = shuffle_dataset(dataset)
                logging.info(f"#Run{self.run_id} Epoch = {epoch}, The size of part, {afile}, = {len(dataset)}")
                training_losses.append(self.run_session(dataset, 'training'))
            if epoch <= 0:
                self.overfit = not self.overfit
            else:
                self.overfit = False
            logging.info(f'#Run{self.run_id} Total Average Training Loss in Epoch {epoch} is '
                         f'{str(np.mean(training_losses))}')
            self.save_model()
            validation_losses = []
            for afile in validation_files:
                with open(afile, 'rb') as handle:
                    dataset = pickle.load(handle)
                logging.info(f"#Run{self.run_id} Epoch = {epoch}, The size of part, {afile}, = {len(dataset)}")
                validation_losses.append(self.run_session(dataset, 'validation'))
            logging.info(f'#Run{self.run_id} Total Average Validation Loss in Epoch {epoch} is '
                         f'{str(np.mean(validation_losses))}')


def train_models_for_experiment1():
    num_runs = configuration.num_runs
    for i in range(0, num_runs):
        obj = PathLength2Training(i)
        obj.train_model()


if __name__ == '__main__':
    initialization()
    train_models_for_experiment1()
