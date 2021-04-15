import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from config import SAConfig
from transformers import AutoTokenizer, AutoModel


class Model:
    def __init__(self):
        pass

    def predict(self, x, batch_size):
        pass


class BertModelSA(nn.Module):
    def __init__(self, bert, tokenizer, config=SAConfig(), max_length=128):
        super().__init__()

        self.bert = bert
        self.tokenizer = tokenizer
        self.config = config

        self.hidden = 768
        self.output_dim = self.config.num_labels
        self.linear = nn.Linear(self.hidden, self.output_dim)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.max_length = max_length

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))

    def forward(self, x, ):
        input_ids = self.tokenizer(x, truncation=True, return_tensors='pt',
                              max_length=self.max_length).to(self.config.device)
        bert_outputs = self.bert(**input_ids)

        outputs = bert_outputs.last_hidden_state.unsqueeze(1)
        outputs = F.avg_pool2d(outputs, (outputs.shape[2], 1)).squeeze(1).squeeze(1)

        # outputs = self.dropout(bert_outputs.pooler_output)

        # outputs = bert_outputs.last_hidden_state[:, 0]

        out = self.linear(outputs)
        out = self.log_softmax(out)
        return out

    def forward_and_get_loss(self, x, y, criterion):
        out = self.forward(x)
        y = torch.LongTensor(y).to(self.config.device)
        return out, criterion(out, y)

    def setup_new_model(model):
        model.to(model.config.device)
        model.train()
        return model


