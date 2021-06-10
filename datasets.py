import json
import os
from functools import lru_cache

import h5py
import torch
from torch.utils.data import Dataset


@lru_cache()
def on_debug():
    return os.environ.get("DEBUG", "") == "1"


def load_json(path_):
    with open(path_, 'r') as j:
        return json.load(j)


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}, self.split
        self._data_folder = data_folder
        self._data_name = data_name

        # Load encoded captions (completely into memory)
        caption_path = os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json')
        caption_lenght_path = os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json')
        self.captions = load_json(caption_path)
        self.caplens = load_json(caption_lenght_path)

        self.transform = transform
        self.dataset_size = len(self.captions)

    def __getitem__(self, index):
        # Open hdf5 file where images are stored
        hdf5_path = os.path.join(self._data_folder, self.split + '_IMAGES_' + self._data_name + '.hdf5')
        with h5py.File(hdf5_path, 'r') as h:
            image_set = h['images']
            # Captions per image
            cpi = h.attrs['captions_per_image']

            # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
            img = torch.from_numpy(image_set[index // cpi] / 255.).float()
            if self.transform is not None:
                img = self.transform(img)

            caption = torch.LongTensor(self.captions[index])
            caplen = torch.LongTensor([self.caplens[index]])

            if self.split == 'TRAIN':
                return img, caption, caplen

            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((index // cpi) * cpi):(((index // cpi) * cpi) + cpi)]
            )
            return img, caption, caplen, all_captions

    def __len__(self):
        if on_debug():
            return self.dataset_size // 20
        return self.dataset_size

# from torch.nn.utils.rnn import pad_sequence
# class PadCollate:
#     """
#     a variant of callate_fn that pads according to the longest sequence in
#     a batch of sequences
#     """
#
#     def __init__(self, dim=0):
#         """
#         args:
#             dim - the dimension to be padded (dimension of time in sequences)
#         """
#         self.dim = dim
#
#     def pad_collate(self, batch):
#         """
#         args:
#             batch - list of (tensor, label)
#
#         reutrn:
#             xs - a tensor of all examples in 'batch' after padding
#             ys - a LongTensor of all labels in batch
#         """
#         # find longest sequence
#         max_len = max(map(lambda x: x[0].shape[self.dim], batch))
#         # pad according to max_len
#         batch = map(lambda (x, y):
#                     (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
#         # stack all
#         xs = torch.stack(map(lambda x: x[0], batch), dim=0)
#         ys = torch.LongTensor(map(lambda x: x[1], batch))
#         return xs, ys
#
#     def __call__(self, batch):
#         return self.pad_collate(batch)
