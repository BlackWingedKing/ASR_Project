# reference:    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
import torchaudio
import os
# from skimage import io
import skimage
import skvideo.io

class DataLoader(Dataset):
    def __init__(self, train_list, path='data/train/', transform=None):
        self.path = path
        self.list = train_list
        self.transform = transform

    def __getitem__(self, index):
        # vid, aud_unshifted, info = io.read_video(self.path+'snippet/vid_'+str(index)+'.mp4')

        vid = torch.Tensor(skvideo.io.vread(self.path+'snippet/' + self.list[index]))
        aud_unshifted, info = torchaudio.load(self.path+'unshifted/' + self.list[index][:-4]+'.wav')
        aud_shifted, info = torchaudio.load(self.path+'shifted/' + self.list[index][:-4]+'.wav')

        # normalising video to -1 and 1
        vid = (2./255)*vid.double() - 1
        # normalising audio
        aud_shifted = normalize_sfs(aud_shifted)
        aud_unshifted = normalize_sfs(aud_unshifted)

        vid = vid.permute(3,0,1,2) # since skvideo reads (T,H,W,C) and our model needs
        # (B, C, T, H, W)
        if self.transforms is not None:
            vid = self.transform(vid)
        return (vid, aud_shifted, aud_unshifted)

    def __len__(self):
        return len(self.list)  # number of videos


def normalize_sfs(sfs, scale=255.):
    return torch.sign(sfs)*(torch.log(1 + scale*torch.abs(sfs)) / torch.log(1 + scale))

class Resize(object):
    """Resize the video to a given size.

    Args:
        output_size: Desired output size. Should be of the form [Time,Height,Width,Channel]
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
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


class RandomCrop(object):
    """Randomly crop out a patch from video of given size

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, vid):
        dim = vid.shape
        new_dim = self.output_size
        assert dim[1] > new_dim[0] and dim[2] > new_dim[1]

        top = torch.randint(dim[1] - new_dim[0],size=(1,1))[0,0]
        left = torch.randint(dim[2] - new_dim[1],size=(1,1))[0,0]

        vid_list = list(map(lambda x: x[top:top+new_dim[0], left:left+new_dim[1],:].reshape(1,new_dim[0],new_dim[1],dim[3]), vid))
        crop_vid = torch.FloatTensor(vid_list)

        return crop_vid
