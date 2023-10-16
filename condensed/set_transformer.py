import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from sklearn.preprocessing import normalize

from reasoning_with_vectors.conf import configuration


def arrange_data_for_set_transformer(xzy, segment_ids):
    unique, counts = np.unique(segment_ids, return_counts=True)
    max_count = np.max(counts)
    num_unique_classes = len(unique)
    xzy_arranged = np.zeros((num_unique_classes, max_count, xzy.shape[1]))
    for i, segment_id in enumerate(unique):
        xzy_arranged[i, :counts[i]] = xzy[segment_ids == segment_id]
    return xzy_arranged


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V, bias=False)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=False)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=False)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=False)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        m_Q = ((Q != 0).any(dim=-1, keepdim=True)).float().expand_as(Q)
        m_K = ((K != 0).any(dim=-1, keepdim=True)).float().expand_as(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        m_Q = torch.cat(m_Q.split(dim_split, 2), 0)
        m_K = torch.cat(m_K.split(dim_split, 2), 0)
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        mask = m_Q.bmm(m_K.transpose(1, 2)) / math.sqrt(self.dim_V)
        A = A.masked_fill(mask == 0, float('-inf'))
        A = torch.softmax(A, 2)
        A = torch.nan_to_num(A, nan=0.0)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, dim_input=2 * configuration.vector_space_dimension,
                 num_outputs=1,
                 dim_output=configuration.vector_space_dimension,
                 num_inds=configuration.num_inds,
                 dim_hidden=configuration.inner_layer_dimension,
                 num_heads=configuration.num_heads,
                 ln=True):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln))
        # SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        # ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
        # ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))
        self.model_file = os.path.join(configuration.model_save_path, f'{SetTransformer.__name__}_'
                                                                      f'{configuration.inner_layer_dimension}.pth')
        self.loss = None
        self.optimizer = None
        self.device = None
        self.init_model()

    def init_model(self):
        self.loss = nn.CosineSimilarity(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=configuration.learning_rate)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_gpus = torch.cuda.device_count()
        if self.device == 'cuda':
            logging.info(f"There are {num_gpus} GPUs:")
            for gpu_number in range(num_gpus):
                logging.info(f"  GPU {gpu_number}: {torch.cuda.get_device_name(gpu_number)}")
            logging.info(f"SetTransformer model currently can not utilize mutiple GPUs. "
                         f"So, it is running on one GPU\n")
        else:
            logging.info("SetTransformer model is running on CPU\n")
        self.to(self.device)
        if os.path.exists(self.model_file):
            self.load_model()
            logging.info(f"Loaded model weights from {self.model_file}")
        else:
            logging.info("Model file does not exist. Initializing weights randomly.")

    def forward(self, X):
        return self.dec(self.enc(X))

    def save_model(self):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_file)

    def load_model(self):
        checkpoint = torch.load(self.model_file, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def run_training(self, xz_list, zy_list, segmentid_list, xy_list, num_iterations=1):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        loss = None
        xz = normalize(np.vstack(xz_list))
        zy = normalize(np.vstack(zy_list))
        xzy = np.concatenate((xz, zy), axis=1)
        segment_ids = np.hstack(segmentid_list)
        xzy = arrange_data_for_set_transformer(xzy, segment_ids)
        xzy = torch.tensor(xzy.astype('float32')).to(self.device)
        xy = torch.tensor(normalize(np.vstack(xy_list)).astype('float32')).to(self.device)
        for i in range(num_iterations):
            self.optimizer.zero_grad()
            xy_pred = self(xzy).squeeze(1)
            loss = 1 - self.loss(xy_pred, xy).mean()
            loss.backward()
            self.optimizer.step()
        final_loss = loss.item()
        logging.info("Training loss = %s, Number of concept pairs = %d", str(final_loss), len(xy_list))
        return final_loss

    def run_prediction(self, xz_list, zy_list, segmentid_list):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        xz = normalize(np.vstack(xz_list))
        zy = normalize(np.vstack(zy_list))
        xzy = np.concatenate((xz, zy), axis=1)
        segment_ids = np.hstack(segmentid_list)
        xzy = arrange_data_for_set_transformer(xzy, segment_ids)
        xzy = torch.tensor(xzy.astype('float32')).to(self.device)
        with torch.no_grad():
            xy_pred = self(xzy).squeeze(1)
        return xy_pred.cpu().tolist()

    def run_validation(self, xz_list, zy_list, segmentid_list, xy_list):
        r"""The list segmentid_list should contain increasing whole numbers.
        """
        xz = normalize(np.vstack(xz_list))
        zy = normalize(np.vstack(zy_list))
        xzy = np.concatenate((xz, zy), axis=1)
        segment_ids = np.hstack(segmentid_list)
        xzy = arrange_data_for_set_transformer(xzy, segment_ids)
        xzy = torch.tensor(xzy.astype('float32')).to(self.device)
        xy = torch.tensor(normalize(np.vstack(xy_list)).astype('float32')).to(self.device)
        with torch.no_grad():
            xy_pred = self(xzy).squeeze(1)
            validation_loss = 1 - self.loss(xy_pred, xy).mean()
        validation_loss = validation_loss.item()
        logging.info("Validation loss = %s, Number of concept pairs = %d", str(validation_loss), len(xy_list))
        return validation_loss


if __name__ == '__main__':
    set_transformer = SetTransformer(dim_input=6, num_outputs=1, dim_output=6,
                                     num_inds=2, dim_hidden=3, num_heads=3, ln=True)
    xz_list_test = [[[1, 2, 3]], [[3, 2, 1], [1, 2, 5]], [[2, 1, 1], [9, 8, 2], [1, 1, 2]]]
    zy_list_test = [[[2, 6, 1]], [[1, 2, 1], [2, 2, 4]], [[5, 1, 3], [3, 2, 1], [3, 3, 3]]]
    segmentid_list_test = [0, 2, 2, 3, 3, 3]
    xy_list_test = [[1, 2, 3, 2, 6, 1], [2, 2, 3, 1.5, 2, 2.5], [4, 3.34, 1.67, 3.67, 2, 2.34]]
    loss = set_transformer.run_training(xz_list_test, zy_list_test, segmentid_list_test, xy_list_test, 100)
    print(loss)
