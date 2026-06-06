import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

class SegmentationLoss:
    def __init__(self, **kargs):
        pass

    def __call__(self, target, output):
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)

        matching_indices = hungarian_matching(output, target)
        output = batch_reordering(output, matching_indices)

        loss = torch.mean(
            torch.sum(
                -(target * torch.log(output) + (1 - target) * torch.log(1 - output)),
                dim=2,
            )
        )

        return loss

def hungarian_matching(W_pred, W_gt):
    """
    This non-tf function does not backprob gradient, only output matching indices
    W_pred - B x N x K
    W_gt: one-hot encoding of I_gt - B x N, may contain -1's
    Output: matching_indices - B x K, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
            where only n_gt_labels entries on each row have meaning. The matching does not include gt background instance
    """
    W_pred = W_pred.detach().cpu().numpy()
    W_gt = W_gt.detach().cpu().numpy()

    batch_size = W_pred.shape[0]
    n_max_labels = W_pred.shape[2]

    matching_indices = np.zeros([batch_size, n_max_labels], dtype=np.int32)
    for b in range(batch_size):
        dot = np.sum(
            np.expand_dims(W_gt[b], axis=2) * np.expand_dims(W_pred[b], axis=1), axis=0
        )
        denominator = (
            np.expand_dims(np.sum(W_gt[b], axis=0), axis=1)
            + np.expand_dims(np.sum(W_pred[b], axis=0), axis=0)
            - dot
        )
        cost = dot / np.maximum(denominator, 1e-10)

        _, col_ind = linear_sum_assignment(-cost)
        matching_indices[b, :] = col_ind

    return matching_indices

def batch_reordering(pred, matching_indices):

    num_batch = pred.shape[0]
    reordering_mat = np.zeros(
        (num_batch, matching_indices.shape[1], matching_indices.shape[1])
    )
    for batch in range(num_batch):
        reordering_mat[batch] = np.linalg.inv(
            np.eye(matching_indices[batch].shape[0])[matching_indices[batch]]
        )

    reordering_mat = torch.tensor(
        reordering_mat, dtype=torch.float, requires_grad=False
    ).to(pred.get_device())
    pred = torch.matmul(pred, reordering_mat)

    return pred
