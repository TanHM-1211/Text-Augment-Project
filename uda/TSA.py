"""
Reimplement Training Signal Annealing in google-research/UDA
"""

import torch
import numpy as np


class TSASchedule:
    def __init__(self, T: int, beta: float):
        # total number of training steps
        self.T = T
        self.beta = beta

    def __call__(self, t, *args, **kwargs):
        pass


class LinearSchedule(TSASchedule):
    def __call__(self, t, *args, **kwargs):
        return t/self.T


class LogSchedule(TSASchedule):
    def __call__(self, t, *args, **kwargs):
        return 1 - np.e ** (- self.beta * t/self.T)


class ExpSchedule(TSASchedule):
    def __call__(self, t, *args, **kwargs):
        return np.e ** (self.beta * (t/self.T - 1))


class TSA:
    def __init__(self, T, K, alpha_t=ExpSchedule, beta=5.0):
        """

        :param T: total number of training steps
        :param K: number of categories
        :param alpha_t: scheduler for increasing threshold
        :param beta: for normalize (?) scheduler
        """

        self.T = T
        self.K = K
        self.alpha_t = alpha_t(self.T, beta)

    def __call__(self, t):
        """

        :param t: step
        :return:
        """
        return self.alpha_t(t) * (1 - 1 / self.K) + 1 / self.K


class TSA_CrossEntropyLoss:
    def __init__(self, tsa: TSA):
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.tsa = tsa

        self.current_step = 0

    def get_mask(self, logits: torch.Tensor, targets: torch.Tensor, current_step):
        logits = torch.log_softmax(logits, dim=-1).detach()
        max_values, pred = torch.max(logits, dim=-1)

        wrong_pred = (torch.abs(pred - targets)>0)
        mask = (max_values < self.tsa(current_step))
        # print(wrong_pred)
        # print(mask)
        mask |= wrong_pred

        mask.to(logits.device)
        return mask

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, current_step=None):
        if current_step is None:
            current_step = self.current_step
        mask = self.get_mask(logits, targets, current_step)
        loss_value = self.loss_func(logits, targets) * mask
        loss_value = loss_value.sum() / max(sum(mask.tolist()), 1)

        self.current_step += 1
        return loss_value

# criterion = TSA_CrossEntropyLoss(TSA(T=num_data/batch_size * epochs, K=2))

