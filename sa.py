import numpy as np
import torch
import os
import time

from model import BertModelSA

SAVE_DIR = './'
device = 'cuda'


def torch_save(dir, model, optimizer, scheduler, all_train_loss=None, all_dev_loss=None,
               best_dev_loss=0., best_dev_acc=1.):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'all_train_loss': all_train_loss,
        'all_dev_loss': all_dev_loss,
        'best_dev_loss': best_dev_loss,
        'best_dev_acc': best_dev_acc,
    }, dir)


def torch_load(dir, model, optimizer, scheduler):
    checkpoint = torch.load(dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    all_train_loss = checkpoint['all_train_loss']
    all_dev_loss = checkpoint['all_dev_loss']
    best_dev_loss = checkpoint['best_dev_loss']
    best_dev_acc = checkpoint['best_dev_acc']

    return all_train_loss, all_dev_loss, best_dev_loss, best_dev_acc


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, x, y, x_valid, y_valid, epochs=num_epochs, batch_size=batch_size, criterion=criterion,
          print_freq=0.05, end_warmup_epoch=1, save_dir=SAVE_DIR, verbose=1):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    best_dev_loss = 1e9
    best_dev_acc = 0
    model.train()

    print_after = int(print_freq * len(x))
    start_time = time.time()
    all_train_loss = []
    all_dev_loss = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7, )
    for epoch in range(epochs):
        if epoch == end_warmup_epoch:
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        print_counter = 0
        total_loss = []
        print('epoch:', epoch)
        for i in tqdm(range(0, len(x), batch_size)):
            prob, loss = model.forward_and_get_loss(x[i: i + batch_size],
                                                    y[i: i + batch_size],
                                                    criterion
                                                    )
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss.append(loss.item())
            if i > print_counter and verbose == 1:
                print('step: {}, loss: {}, total loss: {}'.format(i, loss.item(), np.mean(total_loss)))
                print_counter += print_after
        if epoch >= end_warmup_epoch:
            scheduler.step()
        # print('lr: ', get_lr(optimizer))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_{}.pt'.format(str(epoch))))
        print('train loss:', np.mean(total_loss))

        dev_loss, dev_acc = eval(model, x_valid, y_valid, criterion=criterion)
        print('dev_loss:', dev_loss)

        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best-model-loss.pt'))
            best_dev_loss = dev_loss
        if dev_acc > best_dev_acc:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best-model-acc.pt'))
            best_dev_acc = dev_acc

        all_train_loss.append(total_loss)
        all_dev_loss.append(dev_loss)
        end_time = time.time()
        print('Finish epoch {} at {}, in {} seconds. \n'.format(epoch, end_time, end_time - start_time))

    return model


def eval(model, x, y, batch_size=batch_size, full_detail=True, criterion=criterion, get_confusion=False,
         get_report=False):
    model.eval()
    total_loss = []
    y_pred = []
    y_true = y

    with torch.no_grad():
        for i in tqdm(range(0, len(x), batch_size)):
            prob, loss = model.forward_and_get_loss(x[i: i + batch_size],
                                                    y[i: i + batch_size],
                                                    criterion
                                                    )

            total_loss.append(loss.item())
            y_pred.extend(torch.argmax(prob, dim=-1).tolist())

    report = classification_report(y_true, y_pred, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    if full_detail:
        print(report['macro avg'])
        print(report['weighted avg'])
        print(report)
        # print(confusion)
    if get_report:
        return report
    if get_confusion:
        return confusion, report['accuracy']

    model.train()
    return np.mean(total_loss), report['accuracy']


def test_models(x, y, x_valid, y_valid, x_test, y_test, make_model_func=make_new_model, num_test=3, train_func=train,
                criterion=criterion,
                epochs=num_epochs, batch_size=batch_size, print_freq=0.05, end_warmup_epoch=1,
                save_dir='/content/savetest', verbose=0):
    res = {'accuracy': [], 'weighted avg': [], '0': [], '1': [], '2': []}
    for i in range(num_test):
        print('\n******** TEST {}'.format(i))
        test_model = make_new_model()
        test_model = train_func(test_model, x, y, x_valid, y_valid, epochs=num_epochs, batch_size=batch_size,
                                criterion=criterion,
                                print_freq=print_freq, end_warmup_epoch=end_warmup_epoch, save_dir=save_dir,
                                verbose=verbose)
        test_model.load_state_dict(torch.load(os.path.join(save_dir, 'best-model-acc.pt')))
        test_model.to(device)
        test_model.eval()

        report = eval(test_model, x_test, y_test, criterion=criterion, get_report=True)
        for k in res:
            if k != 'accuracy':
                res[k].append(report[k]['f1-score'])
            else:
                res[k].append(report[k])
    res = {i: np.mean(res[i]) for i in res}
    for i in res:
        if i in reverse_mapping:
            res[reverse_mapping[i]] = res[i]
            del res[i]

    return res
