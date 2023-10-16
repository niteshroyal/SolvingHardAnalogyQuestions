import logging
import os
import pickle

import numpy as np

from reasoning_with_vectors.conf import configuration
# from reasoning_with_vectors.condensed.complex_composition import ComplexCompositionModel
# from reasoning_with_vectors.condensed.composition_with_inverse import CompositionModelInverse
from reasoning_with_vectors.condensed.composition import CompositionModel
# from reasoning_with_vectors.condensed.set_transformer import SetTransformer


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=log_file, filemode='w', level=logging.INFO)
    with open(configuration.__file__, 'r') as f:
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


class PathLength2Training(CompositionModel):
    def __init__(self):
        super().__init__()
        self.overfit = False

    # def run_session(self, dataset, session='training'):
    #     losses = []
    #     xz_list1 = []
    #     zy_list1 = []
    #     xz_list2 = []
    #     zy_list2 = []
    #     segmentid_list = []
    #     xy_list = []
    #     segmentid = -1
    #     for [_, _, xy, xzy] in dataset:
    #         segmentid += 1
    #         xy_list.append(xy)
    #         for [_, xz1, zy1, xz2, zy2] in xzy:
    #             xz_list1.append(xz1)
    #             zy_list1.append(zy1)
    #             xz_list2.append(xz2)
    #             zy_list2.append(zy2)
    #             segmentid_list.append(segmentid)
    #         if len(segmentid_list) >= configuration.training_batch_size:
    #             if session == 'training':
    #                 if self.overfit:
    #                     losses.append(self.run_training(xz_list1, zy_list1, segmentid_list, xy_list, 10,
    #                                                     xz_list2, zy_list2))
    #                 else:
    #                     losses.append(self.run_training(xz_list1, zy_list1, segmentid_list, xy_list, 1,
    #                                                     xz_list2, zy_list2))
    #             else:
    #                 losses.append(self.run_validation(xz_list1, zy_list1, segmentid_list, xy_list, xz_list2, zy_list2))
    #             xz_list1 = []
    #             zy_list1 = []
    #             xz_list2 = []
    #             zy_list2 = []
    #             segmentid_list = []
    #             xy_list = []
    #             segmentid = -1
    #     if len(segmentid_list) > 0:
    #         if session == 'training':
    #             if self.overfit:
    #                 losses.append(self.run_training(xz_list1, zy_list1, segmentid_list, xy_list, 10,
    #                                                 xz_list2, zy_list2))
    #             else:
    #                 losses.append(self.run_training(xz_list1, zy_list1, segmentid_list, xy_list, 1,
    #                                                 xz_list2, zy_list2))
    #         else:
    #             losses.append(self.run_validation(xz_list1, zy_list1, segmentid_list, xy_list,
    #                                               xz_list2, zy_list2))
    #     return np.mean(losses)

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
        logging.info("Number of training data file parts = %d", len(training_files))
        for epoch in range(1, configuration.number_of_epochs + 1):
            training_files = shuffle_files(training_files)
            training_losses = []
            for afile in training_files:
                with open(afile, 'rb') as handle:
                    dataset = pickle.load(handle)
                dataset = shuffle_dataset(dataset)
                logging.info("Epoch = %d, The size of part, %s, = %d", epoch, afile, len(dataset))
                training_losses.append(self.run_session(dataset, 'training'))
            if epoch <= 0:
                self.overfit = not self.overfit
            else:
                self.overfit = False
            logging.info('Total Average Training Loss in Epoch %d is %s', epoch, str(np.mean(training_losses)))
            self.save_model()
            validation_losses = []
            for afile in validation_files:
                with open(afile, 'rb') as handle:
                    dataset = pickle.load(handle)
                logging.info("Epoch = %d, The size of part, %s, = %d", epoch, afile, len(dataset))
                validation_losses.append(self.run_session(dataset, 'validation'))
            logging.info('Total Average Validation Loss in Epoch %d is %s', epoch, str(np.mean(validation_losses)))


if __name__ == '__main__':
    initialization()
    obj = PathLength2Training()
    obj.train_model()
