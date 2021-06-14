import json
import os
import random
from contextlib import contextmanager
from functools import lru_cache

import h5py
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader


@lru_cache()
def on_debug():
    return os.environ.get("DEBUG", "") == "1"


def load_json(path_):
    with open(path_, 'r') as j:
        return json.load(j)


@contextmanager
def set_random_seed(seed):
    preivous_state = random.getstate()
    random.seed(seed)
    yield
    random.setstate(preivous_state)


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
            return self.dataset_size // 50
        return self.dataset_size


class CaptionDataset2(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None, force_output_all_caption=False):
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
        self._force_output_all_caption = force_output_all_caption

        # Load encoded captions (completely into memory)
        caption_path = os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json')
        caption_lenght_path = os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json')
        self.captions = load_json(caption_path)
        self.caplens = load_json(caption_lenght_path)
        hdf5_path = os.path.join(self._data_folder, self.split + '_IMAGES_' + self._data_name + '.hdf5')

        with h5py.File(hdf5_path, 'r') as h:
            # Captions per image
            self._cpi = h.attrs['captions_per_image']
        assert len(self.captions) % self._cpi == 0
        self.transform = transform
        self.dataset_size = int(len(self.captions) / self._cpi)
        if on_debug():
            logger.debug(f"debug mode detected")

    def __getitem__(self, index):
        # Open hdf5 file where images are stored
        hdf5_path = os.path.join(self._data_folder, self.split + '_IMAGES_' + self._data_name + '.hdf5')
        with h5py.File(hdf5_path, 'r') as h:
            image_set = h['images']

            # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
            img = torch.from_numpy(image_set[index] / 255.).float()
            if self.transform is not None:
                img = self.transform(img)

            all_captions = torch.LongTensor(
                self.captions[(index * self._cpi):((index + 1) * self._cpi)]
            )
            seed = random.randint(0, 100)
            with set_random_seed(seed):
                single_caption = random.choice(all_captions)
            with set_random_seed(seed):
                caplen = random.choice(self.caplens[(index * self._cpi):((index + 1) * self._cpi)])
            caption = torch.LongTensor(single_caption)
            caplen = torch.LongTensor([caplen])
            if self._force_output_all_caption:
                return img, caption, caplen, all_captions

            if self.split == 'TRAIN':
                return img, caption, caplen
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            return img, caption, caplen, all_captions

    def __len__(self):
        if on_debug():
            return self.dataset_size // 50
        return self.dataset_size


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """

    def __call__(self, batch):
        batch_size = len(batch)
        img, caption, caplen, *all_captions = list(zip(*batch))
        tensor_image = torch.stack(img, dim=0)
        max_length = 0
        for c in caption:
            max_length = max(max_length, len([x for x in c if x != 0]))
        new_caption = torch.zeros(batch_size, max_length + 1, dtype=torch.long)
        for i, c in enumerate(caption):
            cur_c = torch.Tensor([x for x in c if x != 0])
            new_caption[i][:len(cur_c)] = cur_c

        new_caplen = torch.Tensor(caplen).long()

        return tensor_image, new_caption, new_caplen.unsqueeze(1), *all_captions


if __name__ == '__main__':
    dataset1 = CaptionDataset(data_folder="/opt/dataset/output", data_name='coco_5_cap_per_img_5_min_word_freq',
                              split="TRAIN", transform=lambda x: x, )
    dataset2 = CaptionDataset2(data_folder="/opt/dataset/output", data_name='coco_5_cap_per_img_5_min_word_freq',
                               split="TRAIN", transform=lambda x: x, )
    tra_loader = DataLoader(dataset2, batch_size=4, shuffle=True, collate_fn=PadCollate())
    print(next(iter(tra_loader)))
