class Config:
    device = 'cuda'
    num_epochs = 6
    batch_size = 32
    print_freq = 0.05
    end_warmup_epoch = 1


class SAConfig(Config):
    num_labels = 2
