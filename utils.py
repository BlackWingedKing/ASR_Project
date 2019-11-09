from torch.utils.data import random_split


def split_data(list_vid, train_frac, val_frac):
    len_data = len(list_vid)
    train_size = round(len_data*train_frac)
    val_size = round(len_data*val_frac)
    train_vid, val_vid = random_split(list_vid, [train_size, val_size])
    train_list = list(train_vid)
    val_list = list(val_vid)
    return train_list, val_list


def log_list(l, fname):
    file = open(fname, 'w')
    for i in l:
        file.write(i+'\n')
    file.close()


def read_list(path):
    file = open(path, 'r')
    list_l = [line.split()[0] for line in file.readlines()]
    return list_l
