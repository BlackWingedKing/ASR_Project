# reference:    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from torch.utils.data.dataset import Dataset
import torch
from torchvision import io
import os
import skimage


class DataLoaderTrain(Dataset):
    def __init__(self, path='data/train/', transform=None):
        self.path = path
        self.list = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, index):
        vid, aud, info = io.read_video(self.path+'vid_'+str(index)+'.mp4')
        if len(aud.shape) > 1 and aud.shape[0] > 1:
            aud = aud[0, :]
        if self.transforms is not None:
            vid = self.transform(vid)
        return (vid, aud)

    def __len__(self):
        return len(self.list)  # number of videos


class Resize(object):
    """Resize the video to a given size.

    Args:
        output_size: Desired output size. Should be of the form [Time,Height,Width,Channel]
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, vid):
        vid_resized = torch.zeros(self.output_size)
        # for i in range(dim[0]):
        #     vid_resized[i, :, :, :] = skimage.transform.resize(
        #         vid[i, :, :, :], (self.output_size[1], self.output_size[2], 3))
        vid_list = list(map(lambda x: skimage.transform.resize(
            x, (self.output_size[1], self.output_size[2], 3)), vid))
        vid_resized = torch.FloatTensor(vid_list)
        return vid_resized


class RandomCrop(object):  # INCOMPLETE
    """Randomly crop out a patch from video of given size

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

    def __call__(self, vid):
        dim = vid.shape
        new_dim = self.output_size
        assert dim[0] > new_dim[0] and dim[1] > new_dim[1]

        top = torch.randint(0, dim[0] - new_dim[0])
        left = torch.randint(0, dim[1] - new_dim[1])

        vid_list = list(map(lambda x: x[top:top+new_dim[0], left:left+new_dim[1]], vid))
        crop_vid = torch.FloatTensor(vid_list)

        return crop_vid
