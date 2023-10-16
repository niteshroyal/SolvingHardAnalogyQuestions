import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import normalize

from reasoning_with_vectors.conf import configuration


def arrange_segment_ids_in_order(segmentid_list):
    _, ordered = np.unique(segmentid_list, return_inverse=True)
    return list(ordered)


class ComplexCompositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector_space_dimension = configuration.vector_space_dimension
        self.inner_layer_dimension = configuration.inner_layer_dimension
        self.output_layer_dimension = configuration.vector_space_dimension
        self.model_file = os.path.join(configuration.model_save_path,
                                       ComplexCompositionModel.__name__ +
                                       str(configuration.importance_threshold) + '.pth')
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None
        self.w5 = None
        self.w6 = None
        self.gelu = None
        self.init_all_weights()
        self.loss = None
        self.optimizer = None
        self.device = None
        self.init_model()

    def init_all_weights(self):
        self.w1 = nn.Linear(2 * self.vector_space_dimension, self.inner_layer_dimension)
        self.w2 = nn.Linear(self.vector_space_dimension, self.inner_layer_dimension)
        self.w3 = nn.Linear(self.vector_space_dimension, self.inner_layer_dimension)
        self.w4 = nn.Linear(self.vector_space_dimension, self.inner_layer_dimension)
        self.w5 = nn.Linear(self.vector_space_dimension, self.inner_layer_dimension)
        self.w6 = nn.Linear(3 * self.inner_layer_dimension, self.output_layer_dimension)
        self.gelu = nn.GELU()

    def init_model(self):
        self.loss = nn.CosineSimilarity(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=configuration.learning_rate)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_gpus = torch.cuda.device_count()
        if self.device == 'cuda':
            logging.info(f"There are {num_gpus} GPUs:")
            for gpu_number in range(num_gpus):
                logging.info(f"  GPU {gpu_number}: {torch.cuda.get_device_name(gpu_number)}")
            logging.info(f"ComplexCompositionModel currently can not utilize multiple GPUs. "
                         f"So, it is running on one GPU\n")
        else:
            logging.info("ComplexCompositionModel is running on CPU\n")
        self.to(self.device)
        if os.path.exists(self.model_file):
            self.load_model()
            logging.info(f"Loaded model weights from {self.model_file}")
        else:
            logging.info("Model file does not exist. Initializing weights randomly.")

    def forward(self, xz, zy, segment_ids):
        m = segment_ids[-1].item() + 1
        xy = torch.cat([xz, zy], dim=1)
        a = self.gelu(self.w1(xy))
        b = self.gelu(self.w2(xz))
        b = b * self.gelu(self.w3(zy))
        c = self.gelu(self.w4(zy))
        c = c * self.gelu(self.w5(xz))
        abc = torch.cat([a, b, c], dim=1)
        n = abc.size(1)
        segment_sum = torch.zeros((m, n)).to(abc.device)
        segment_sum.scatter_add_(0, segment_ids.unsqueeze(-1).expand_as(abc), abc)
        abc = self.w6(segment_sum)
        return abc

    def run_training(self, xz_list, zy_list, segmentid_list, xy_list, num_iterations=1, xz_list2=None, zy_list2=None):
        loss = None
        xz = torch.tensor(normalize(np.vstack(xz_list)).astype('float32')).to(self.device)
        zy = torch.tensor(normalize(np.vstack(zy_list)).astype('float32')).to(self.device)
        segmentid_list = arrange_segment_ids_in_order(segmentid_list)
        segment_ids = torch.tensor(np.hstack(segmentid_list).astype('int64')).to(self.device)
        xy = torch.tensor(normalize(np.vstack(xy_list)).astype('float32')).to(self.device)
        for i in range(num_iterations):
            self.optimizer.zero_grad()
            xy_pred = self(xz, zy, segment_ids)
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

    def run_prediction(self, xz_list, zy_list, segmentid_list, xz_list2=None, zy_list2=None):
        xz = torch.tensor(normalize(np.vstack(xz_list)).astype('float32')).to(self.device)
        zy = torch.tensor(normalize(np.vstack(zy_list)).astype('float32')).to(self.device)
        segmentid_list = arrange_segment_ids_in_order(segmentid_list)
        segment_ids = torch.tensor(np.hstack(segmentid_list).astype('int64')).to(self.device)
        with torch.no_grad():
            xy_pred = self(xz, zy, segment_ids)
        return xy_pred.cpu().tolist()

    def run_validation(self, xz_list, zy_list, segmentid_list, xy_list, xz_list2=None, zy_list2=None):
        xz = torch.tensor(normalize(np.vstack(xz_list)).astype('float32')).to(self.device)
        zy = torch.tensor(normalize(np.vstack(zy_list)).astype('float32')).to(self.device)
        segmentid_list = arrange_segment_ids_in_order(segmentid_list)
        segment_ids = torch.tensor(np.hstack(segmentid_list).astype('int64')).to(self.device)
        xy = torch.tensor(normalize(np.vstack(xy_list)).astype('float32')).to(self.device)
        with torch.no_grad():
            xy_pred = self(xz, zy, segment_ids)
            validation_loss = 1 - self.loss(xy_pred, xy).mean()
        validation_loss = validation_loss.item()
        logging.info("Validation loss = %s, Number of concept pairs = %d", str(validation_loss), len(xy_list))
        return validation_loss

