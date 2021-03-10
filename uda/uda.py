import torch
import numpy as np
import torch.nn.functional as F

from torch import nn


class FLAGS:
    device = 'cpu'
    uda_softmax_temp = 0.4
    uda_confidence_thresh = 0.8
    num_labels = 2
    sup_batch_size = 8
    unsup_batch_size = 16
    num_epochs = 5

def kl_for_log_probs(log_p, log_q):
  p = torch.exp(log_p)
  neg_ent = torch.sum(p * log_p, dim=-1)
  neg_cross_ent = torch.sum(p * log_q, dim=-1)
  kl = neg_ent - neg_cross_ent
  return kl


def forward_and_get_uda_loss(model, x, y, unlabeled_x, unlabeled_x_aug, supervised_loss_func):
    # supervised loss
    out = model.forward(x)
    supervised_loss = supervised_loss_func(out, y)

    # unsupervised loss
    unsup_loss_mask = 1
    pred_unlabeled_x_logits = []
    pred_unlabeled_x_aug_logits = []
    if len(unlabeled_x) == 0 or len(unlabeled_x_aug) == 0:
        return supervised_loss
    for unsup_batch in range(0, len(unlabeled_x), len(x)):
        with torch.no_grad():
            pred_unlabeled_x_logits.append(model.forward(unlabeled_x))

        pred_unlabeled_x_aug_logits.append(model.forward(unlabeled_x_aug))

    pred_unlabeled_x_logits = torch.cat(pred_unlabeled_x_logits, dim=-1)
    pred_unlabeled_x_aug_logits = torch.cat(pred_unlabeled_x_aug_logits, dim=-1)
    pred_unlabeled_x_log_probs = torch.log_softmax(pred_unlabeled_x_logits, dim=-1)
    tgt_pred_unlabeled_x_log_probs = torch.log_softmax(pred_unlabeled_x_logits / FLAGS.uda_softmax_temp, dim=-1)
    pred_unlabeled_x_aug_log_probs = torch.log_softmax(pred_unlabeled_x_aug_logits, dim=-1)

    if FLAGS.uda_confidence_thresh != -1:
        max_values, pred = torch.max(pred_unlabeled_x_log_probs, dim=-1)
        with torch.no_grad():
            unsup_loss_mask = (max_values < FLAGS.uda_confidence_thresh)

    per_example_kl_loss = kl_for_log_probs(tgt_pred_unlabeled_x_log_probs,
                                           pred_unlabeled_x_aug_log_probs) * unsup_loss_mask

    unsupervised_loss = torch.mean(per_example_kl_loss)
    return supervised_loss + unsupervised_loss



