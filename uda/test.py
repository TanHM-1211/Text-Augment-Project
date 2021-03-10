import torch
import time
import numpy as np

from torch import nn
from tqdm import tqdm
from TSA import TSA_CrossEntropyLoss, TSA
from uda import forward_and_get_uda_loss, FLAGS


torch.random.manual_seed(seed=222)

FLAGS.device = 'cpu'
FLAGS.num_labels = 2
FLAGS.sup_batch_size = 8
FLAGS.unsup_batch_size = 16
FLAGS.num_epochs = 5
num_data = 3200


def make_model():
    model = nn.Sequential(
        nn.Linear(10, 200),
        nn.Linear(200, FLAGS.num_labels)
    )
    return model


model = make_model()
model.to(FLAGS.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

x = torch.rand((num_data, 10), device=FLAGS.device)
unlabeled_x = torch.rand((num_data * 2, 10), device=FLAGS.device)
unlabeled_x_aug = torch.rand((num_data * 2, 10), device=FLAGS.device)
y = torch.randint(0, FLAGS.num_labels, (num_data,), dtype=torch.long, device=FLAGS.device)

supervised_loss_func = TSA_CrossEntropyLoss(TSA(T=num_data/FLAGS.sup_batch_size * FLAGS.num_epochs, K=FLAGS.num_labels))


def train_uda(model, x, y, unlabeled_x, unlabeled_x_aug, supervised_loss_func,
              sup_batch_size=FLAGS.sup_batch_size, unsup_batch_size=FLAGS.unsup_batch_size,
              epochs=FLAGS.num_epochs, print_freq=0.05):
    best_dev_loss = 1e9
    model.train()

    print_after = int(print_freq * len(x))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, )
    for epoch in range(epochs):
        print_counter = 0
        total_loss = []
        print('epoch:', epoch)
        for i in tqdm(range(0, len(x), sup_batch_size)):
            num_unsup_batches = i // sup_batch_size
            loss = forward_and_get_uda_loss(model, x[i: i + sup_batch_size], y[i: i + sup_batch_size],
                                            unlabeled_x[num_unsup_batches: num_unsup_batches + unsup_batch_size],
                                            unlabeled_x_aug[num_unsup_batches: num_unsup_batches + unsup_batch_size],
                                            supervised_loss_func)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss.append(loss.item())
            if i > print_counter:
                print('step: {}, loss: {}, total loss: {}'.format(i, loss.item(), np.mean(total_loss)))
                print_counter += print_after
        scheduler.step()
        print('train loss:', np.mean(total_loss))


train_uda(model, x, y, unlabeled_x, unlabeled_x_aug, supervised_loss_func)
# test_criterion(x, y, nn.CrossEntropyLoss())
