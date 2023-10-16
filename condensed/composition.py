import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import normalize
import torch.nn.functional as functional

from reasoning_with_vectors.conf import configuration


def arrange_segment_ids_in_order(segmentid_list):
    _, ordered = np.unique(segmentid_list, return_inverse=True)
    return list(ordered)


class CompositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector_space_dimension = configuration.vector_space_dimension
        self.inner_layer_dimension = configuration.inner_layer_dimension
        self.output_layer_dimension = configuration.vector_space_dimension
        self.model_file = os.path.join(configuration.model_save_path, f'{CompositionModel.__name__}_'
                                                                      f'{configuration.importance_threshold}_'
                                                                      f'{configuration.inner_layer_dimension}.pth')
        self.w1 = None
        self.gelu = None
        self.w2 = None
        self.init_all_weights()
        self.loss = None
        self.optimizer = None
        self.device = None
        self.init_model()

    def init_all_weights(self):
        self.w1 = nn.Linear(2 * self.vector_space_dimension, self.inner_layer_dimension)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(self.inner_layer_dimension, self.output_layer_dimension)

    def init_model(self):
        self.loss = nn.CosineSimilarity(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=configuration.learning_rate)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_gpus = torch.cuda.device_count()
        if self.device == 'cuda':
            logging.info(f"There are {num_gpus} GPUs:")
            for gpu_number in range(num_gpus):
                logging.info(f"  GPU {gpu_number}: {torch.cuda.get_device_name(gpu_number)}")
            logging.info(f"CompositionModel currently can not utilize mutiple GPUs. "
                         f"So, it is running on one GPU\n")
        else:
            logging.info("CompositionModel is running on CPU\n")
        self.to(self.device)
        if os.path.exists(self.model_file):
            self.load_model()
            logging.info(f"Loaded model weights from {self.model_file}")
        else:
            logging.info("Model file does not exist. Initializing weights randomly.")

    def forward(self, xzy, segment_ids):
        r"""The list segment_ids should contain increasing whole numbers, 0 should be its first element,
        and should not contain missing numbers. Here is an example of a valid segment_ids:

        segment_ids = tensor([0, 0, 1, 2, 3, 3, 4, 4, 4, 4])

        Note that tensor([0, 0, 2, 2, 3, 3, 4, 4, 4, 4]) is not a valid segment_ids.
        """
        m = segment_ids[-1].item() + 1
        xy = self.gelu(self.w1(xzy))
        n = xy.size(1)
        segment_sum = torch.zeros((m, n)).to(xy.device)
        segment_sum.scatter_add_(0, segment_ids.unsqueeze(-1).expand_as(xy), xy)
        xy = self.w2(segment_sum)
        xy = functional.normalize(xy, p=2, dim=1)
        return xy

    def compute_mean_max_cosine_similarity(self, xzy1, xzy2):
        xzy1_transformed = self.w2(self.gelu(self.w1(xzy1)))
        xzy2_transformed = self.w2(self.gelu(self.w1(xzy2)))
        xzy1_norm = functional.normalize(xzy1_transformed, p=2, dim=1)
        xzy2_norm = functional.normalize(xzy2_transformed, p=2, dim=1)
        result = (xzy1_norm @ xzy2_norm.t()).max(dim=1)[0].mean()
        return result

    def run_training(self, xz_list, zy_list, segmentid_list, xy_list, num_iterations=1):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        loss = None
        xz = torch.tensor(normalize(np.vstack(xz_list)).astype('float32'))
        zy = torch.tensor(normalize(np.vstack(zy_list)).astype('float32'))
        xzy = torch.cat((xz, zy), dim=1).to(self.device)

        segmentid_list = arrange_segment_ids_in_order(segmentid_list)
        segment_ids = torch.tensor(np.hstack(segmentid_list).astype('int64')).to(self.device)
        xy = torch.tensor(normalize(np.vstack(xy_list)).astype('float32')).to(self.device)
        for i in range(num_iterations):
            self.optimizer.zero_grad()
            xy_pred = self(xzy, segment_ids)
            loss = 1 - self.loss(xy_pred, xy).mean()
            loss.backward()
            self.optimizer.step()
        final_loss = loss.item()
        logging.info("Training loss = %s, Number of concept pairs = %d", str(final_loss), len(xy_list))
        return final_loss

    def save_model(self):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_file)

    def load_model(self):
        checkpoint = torch.load(self.model_file, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def run_prediction(self, xz_list, zy_list, segmentid_list):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        xz = torch.tensor(normalize(np.vstack(xz_list)).astype('float32'))
        zy = torch.tensor(normalize(np.vstack(zy_list)).astype('float32'))
        xzy = torch.cat((xz, zy), dim=1).to(self.device)
        segmentid_list = arrange_segment_ids_in_order(segmentid_list)
        segment_ids = torch.tensor(np.hstack(segmentid_list).astype('int64')).to(self.device)
        with torch.no_grad():
            xy_pred = self(xzy, segment_ids)
        return xy_pred.cpu().tolist()

    def run_path_based_similarity(self, xz_list1, zy_list1, xz_list2, zy_list2):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        xz1 = torch.tensor(normalize(np.vstack(xz_list1)).astype('float32'))
        zy1 = torch.tensor(normalize(np.vstack(zy_list1)).astype('float32'))
        xzy1 = torch.cat((xz1, zy1), dim=1).to(self.device)
        xz2 = torch.tensor(normalize(np.vstack(xz_list2)).astype('float32'))
        zy2 = torch.tensor(normalize(np.vstack(zy_list2)).astype('float32'))
        xzy2 = torch.cat((xz2, zy2), dim=1).to(self.device)
        with torch.no_grad():
            sim = self.compute_mean_max_cosine_similarity(xzy1, xzy2)
        return sim.cpu().item()

    def run_validation(self, xz_list, zy_list, segmentid_list, xy_list):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        xz = torch.tensor(normalize(np.vstack(xz_list)).astype('float32'))
        zy = torch.tensor(normalize(np.vstack(zy_list)).astype('float32'))
        xzy = torch.cat((xz, zy), dim=1).to(self.device)
        segmentid_list = arrange_segment_ids_in_order(segmentid_list)
        segment_ids = torch.tensor(np.hstack(segmentid_list).astype('int64')).to(self.device)
        xy = torch.tensor(normalize(np.vstack(xy_list)).astype('float32')).to(self.device)
        with torch.no_grad():
            xy_pred = self(xzy, segment_ids)
            validation_loss = 1 - self.loss(xy_pred, xy).mean()
        validation_loss = validation_loss.item()
        logging.info("Validation loss = %s, Number of concept pairs = %d", str(validation_loss), len(xy_list))
        return validation_loss
