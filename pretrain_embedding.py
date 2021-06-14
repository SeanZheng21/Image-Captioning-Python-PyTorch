# this work follows the SIMCSE paper.
import argparse

import torchvision.transforms as transforms
from pytorch_metric_learning import losses
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam

from config import *
from datasets import CaptionDataset2 as CaptionDataset, PadCollate
from models import Encoder, DecoderWithAttention, HiddenStateProjector
from train import train_augment
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", required=True, type=str, help="data folder")
parser.add_argument("--enable-scale", action="store_true", default=False, help="enable mixed training")
parser.add_argument("--batch-size", "-s", default=64, type=int, help="batch size")
parser.add_argument("--save_dir", required=True, type=str, help="path to save the checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")
args = parser.parse_args()

data_folder = args.data_folder  # folder with data files saved by create_input_files.py
batch_size = args.batch_size
checkpoint = args.checkpoint  # path to checkpoint, None if none
enable_amp = args.enable_scale  # enable mixed training for a faster speed and less memory usage.
num_batches = 350

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
word_map = load_word_map(word_map_file)

encoder = Encoder()
decoder = DecoderWithAttention(attention_dim=attention_dim,
                               embed_dim=emb_dim,
                               decoder_dim=decoder_dim,
                               vocab_size=len(word_map),
                               dropout=dropout)
decoder_optimizer = RAdam(params=decoder.parameters(),
                          lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=100, )
encoder.fine_tune(False)
scaler = GradScaler(enabled=enable_amp)

# Move to GPU, if available
decoder = decoder.to(device)
encoder = encoder.to(device)
optimizer_to(decoder_optimizer, device)
# Loss function
criterion = losses.SupConLoss()

# Custom data_loaders
train_set = CaptionDataset(data_folder, data_name, 'TRAIN', transform=train_augment(), force_output_all_caption=True)
train_loader = iter(DataLoader(
    train_set, batch_size=batch_size, num_workers=workers, pin_memory=True,
    sampler=InfiniteRandomSampler(train_set, shuffle=True), collate_fn=PadCollate()))

projector = HiddenStateProjector(input_dim=512 * 2, hidden_dim=256, output_dim=128).to(device)

writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb"))


def sample_from_all_captions(all_captions, n=2):
    def random_index(item, n):
        rnd_indices = np.random.choice(len(item), size=n)
        return item[rnd_indices]

    samples = zip(*[random_index(captions, n) for captions in all_captions])
    return [torch.stack(s, dim=0) for s in samples]


n_samples = 2

for epoch in range(100):
    decoder.train()
    loss_meter = AverageMeter()
    indicator = tqdm(range(num_batches), total=num_batches)
    for i, (image, _, caplens, all_captions) in zip(indicator, train_loader):
        cap1, cap2 = sample_from_all_captions(all_captions=all_captions, n=n_samples)
        image, cap1, cap2 = image.to(device), cap1.to(device), cap2.to(device)
        with autocast(enable_amp):
            with torch.no_grad():
                image_encoding = encoder(image)
            scores, caps_sorted, decode_lengths, alphas, sort_ind, (h1, c1), reverted_index1 = \
                decoder(image_encoding, cap1, caplens, return_hidden_state=True)
            *_, (h2, c2), _ = decoder(image_encoding, cap2, caplens, return_hidden_state=True)
            repr1 = torch.cat([h1, c1], dim=1)
            repr2 = torch.cat([h2, c2], dim=1)
            loss = criterion(torch.cat([repr1, repr2], dim=0),
                             labels=torch.LongTensor(list(range(len(repr1))) * n_samples).to(device))
        scaler.scale(loss).backward()
        scaler.step(decoder_optimizer)
        scaler.update()
        decoder_optimizer.zero_grad()
        loss_meter.update(loss.item(), len(image))
        if i % 10 == 0:
            indicator.set_postfix({"loss": loss_meter.avg})
    state = {
        'decoder': decoder.float().state_dict(),
    }
    torch.save(state, os.path.join(args.save_dir, "pretrained.pth"))
    writer.add_scalar("pretrain/loss", loss_meter.avg, global_step=epoch)
    scheduler.step()
