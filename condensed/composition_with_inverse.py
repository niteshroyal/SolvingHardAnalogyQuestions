import os
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import normalize

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.condensed.composition import CompositionModel, arrange_segment_ids_in_order


class CompositionModelInverse(CompositionModel):
    def __init__(self):
        super().__init__()

    def init_all_weights(self):
        if configuration.inverse:
            self.model_file = os.path.join(configuration.model_save_path,
                                           CompositionModelInverse.__name__ +
                                           str(configuration.importance_threshold) + '.pth')
            self.w1 = nn.Linear(4 * self.vector_space_dimension, 2 * self.inner_layer_dimension)
            self.gelu = nn.GELU()
            self.w2 = nn.Linear(2 * self.inner_layer_dimension, self.output_layer_dimension)
        else:
            super().init_all_weights()

    def run_training(self, xz_list1, zy_list1, segmentid_list, xy_list, num_iterations=1, xz_list2=None, zy_list2=None):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        if not configuration.inverse:
            return super().run_training(xz_list1, zy_list1, segmentid_list, xy_list, num_iterations)
        else:
            loss = None
            xz = torch.tensor(normalize(np.vstack(xz_list1)).astype('float32'))
            zy = torch.tensor(normalize(np.vstack(zy_list1)).astype('float32'))
            xzy = torch.cat((xz, zy), dim=1)
            xz = torch.tensor(normalize(np.vstack(xz_list2)).astype('float32'))
            zy = torch.tensor(normalize(np.vstack(zy_list2)).astype('float32'))
            xzy = torch.cat((xzy, xz), dim=1)
            xzy = torch.cat((xzy, zy), dim=1).to(self.device)
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

    def run_prediction(self, xz_list1, zy_list1, segmentid_list, xz_list2=None, zy_list2=None):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        if not configuration.inverse:
            return super().run_prediction(xz_list1, zy_list1, segmentid_list)
        else:
            xz = torch.tensor(normalize(np.vstack(xz_list1)).astype('float32'))
            zy = torch.tensor(normalize(np.vstack(zy_list1)).astype('float32'))
            xzy = torch.cat((xz, zy), dim=1)
            xz = torch.tensor(normalize(np.vstack(xz_list2)).astype('float32'))
            zy = torch.tensor(normalize(np.vstack(zy_list2)).astype('float32'))
            xzy = torch.cat((xzy, xz), dim=1)
            xzy = torch.cat((xzy, zy), dim=1).to(self.device)
            segmentid_list = arrange_segment_ids_in_order(segmentid_list)
            segment_ids = torch.tensor(np.hstack(segmentid_list).astype('int64')).to(self.device)
            with torch.no_grad():
                xy_pred = self(xzy, segment_ids)
            return xy_pred.cpu().tolist()

    def run_validation(self, xz_list1, zy_list1, segmentid_list, xy_list, xz_list2=None, zy_list2=None):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        if not configuration.inverse:
            return super().run_validation(xz_list1, zy_list1, segmentid_list, xy_list)
        else:
            xz = torch.tensor(normalize(np.vstack(xz_list1)).astype('float32'))
            zy = torch.tensor(normalize(np.vstack(zy_list1)).astype('float32'))
            xzy = torch.cat((xz, zy), dim=1)
            xz = torch.tensor(normalize(np.vstack(xz_list2)).astype('float32'))
            zy = torch.tensor(normalize(np.vstack(zy_list2)).astype('float32'))
            xzy = torch.cat((xzy, xz), dim=1)
            xzy = torch.cat((xzy, zy), dim=1).to(self.device)
            segmentid_list = arrange_segment_ids_in_order(segmentid_list)
            segment_ids = torch.tensor(np.hstack(segmentid_list).astype('int64')).to(self.device)
            xy = torch.tensor(normalize(np.vstack(xy_list)).astype('float32')).to(self.device)
            with torch.no_grad():
                xy_pred = self(xzy, segment_ids)
                validation_loss = 1 - self.loss(xy_pred, xy).mean()
            validation_loss = validation_loss.item()
            logging.info("Validation loss = %s, Number of concept pairs = %d", str(validation_loss), len(xy_list))
            return validation_loss
