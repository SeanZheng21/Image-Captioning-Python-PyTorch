import argparse
import time
from contextlib import nullcontext

import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam

from bleu_score import corpus_bleu
from config import *
from datasets import CaptionDataset2 as CaptionDataset, PadCollate
from models import Encoder, DecoderWithAttention
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", required=True, type=str, help="data folder")
parser.add_argument("--enable-scale", action="store_true", default=False, help="enable mixed training")
parser.add_argument("--batch-size", "-s", default=64, type=int, help="batch size")
parser.add_argument("--finetune-encoder", default=False, action="store_true", help="finetune encoder")
parser.add_argument("--save_dir", required=True, type=str, help="path to save the checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")
args = parser.parse_args()

data_folder = args.data_folder  # folder with data files saved by create_input_files.py
batch_size = args.batch_size
fine_tune_encoder = args.finetune_encoder  # fine-tune encoder?
checkpoint = args.checkpoint  # path to checkpoint, None if none
enable_amp = args.enable_scale  # enable mixed training for a faster speed and less memory usage.

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def train_augment():
    return transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1]),
        normalize
    ])


def val_augment():
    return transforms.Compose([
        transforms.CenterCrop(224),
        normalize
    ])


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=dropout)
    encoder_optimizer = RAdam(params=encoder.parameters(), lr=encoder_lr)
    decoder_optimizer = RAdam(params=decoder.parameters(),
                              lr=decoder_lr, weight_decay=1e-5)
    encoder.fine_tune(fine_tune_encoder)
    scaler = GradScaler(enabled=enable_amp)

    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location="cpu")
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']

        decoder.load_state_dict(checkpoint['decoder'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        encoder.load_state_dict(checkpoint['encoder'])
        with IgnoreException(Exception):
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        scaler.load_state_dict(checkpoint["scaler"])
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    optimizer_to(decoder_optimizer, device)
    optimizer_to(encoder_optimizer, device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_set = CaptionDataset(data_folder, data_name, 'TRAIN', transform=train_augment())
    train_loader = iter(torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=workers, pin_memory=True,
        sampler=InfiniteRandomSampler(train_set, shuffle=True), collate_fn=PadCollate()))
    val_set = CaptionDataset(data_folder, data_name, 'VAL', transform=val_augment())
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb"))
    # Epochs
    with writer:
        for epoch in range(start_epoch, epochs):

            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if epochs_since_improvement == 20:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
                adjust_learning_rate(decoder_optimizer, 0.8)
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_optimizer, 0.8)

            writer.add_scalar("tra/encoder_lr", encoder_optimizer.param_groups[0]["lr"], global_step=epoch)
            writer.add_scalar("tra/decoder_lr", decoder_optimizer.param_groups[0]["lr"], global_step=epoch)
            # One epoch's training
            tra_mean_loss, tra_mean_t5acc = train(train_loader=train_loader,
                                                  encoder=encoder,
                                                  decoder=decoder,
                                                  criterion=criterion,
                                                  encoder_optimizer=encoder_optimizer,
                                                  decoder_optimizer=decoder_optimizer,
                                                  epoch=epoch,
                                                  scaler=scaler)
            writer.add_scalar("tra/loss", tra_mean_loss, global_step=epoch)
            writer.add_scalar("tra/top5acc", tra_mean_t5acc, global_step=epoch)

            # One epoch's validation
            bleu1, bleu2, bleu3, recent_bleu4, val_mean_loss, val_mean_t5acc = validate(val_loader=val_loader,
                                                                                        encoder=encoder,
                                                                                        decoder=decoder,
                                                                                        criterion=criterion,
                                                                                        scaler=scaler)

            writer.add_scalar("val/loss", val_mean_loss, global_step=epoch)
            writer.add_scalar("val/top5acc", val_mean_t5acc, global_step=epoch)
            writer.add_scalar("val/b1", bleu1, global_step=epoch)
            writer.add_scalar("val/b2", bleu2, global_step=epoch)
            writer.add_scalar("val/b3", bleu3, global_step=epoch)
            writer.add_scalar("val/b4", recent_bleu4, global_step=epoch)
            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(data_name, args.save_dir, epoch, epochs_since_improvement, encoder, decoder,
                            encoder_optimizer,
                            decoder_optimizer, recent_bleu4, is_best, scaler)


def train(*, train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, scaler):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    :param scaler: automixed training scaler
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in zip(range(len(train_loader)), train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        caplens = caplens.to(device, non_blocking=True)
        with autocast(enabled=enable_amp):
            with torch.no_grad() if not fine_tune_encoder else nullcontext():
                image_encodings = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(image_encodings, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        if fine_tune_encoder:
            encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        scaler.scale(loss).backward()

        # gradient clip following: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        if fine_tune_encoder:
            scaler.unscale_(encoder_optimizer)
        scaler.unscale_(decoder_optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)  # here we change the absolute value to norm
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)  # here we change the absolute value to norm

        if fine_tune_encoder:
            scaler.step(encoder_optimizer)
        scaler.step(decoder_optimizer)

        scaler.update()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        if torch.isnan(loss):
            raise RuntimeError(loss)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    return losses.avg, top5accs.avg


def validate(val_loader, encoder, decoder, criterion, scaler):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            with autocast(enabled=enable_amp):
                # Forward prop.
                if encoder is not None:
                    imgs = encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                # Calculate loss
                loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu1 = corpus_bleu(references, hypotheses, weights=[1, 0, 0, 0])
        bleu2 = corpus_bleu(references, hypotheses, weights=[0.5, 0.5, 0, 0])
        bleu3 = corpus_bleu(references, hypotheses, weights=[1 / 3, 1 / 3, 1 / 3, 0])
        bleu4 = corpus_bleu(references, hypotheses, weights=[1 / 4, 1 / 4, 1 / 4, 1 / 4])

        print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, '
              'BLEU - [{bleu1:.3f} {bleu2:.3f} {bleu3:.3f} {bleu4:.3f}]\n'.format(
            loss=losses, top5=top5accs, bleu4=bleu4, bleu3=bleu3, bleu2=bleu2, bleu1=bleu1))

    return bleu1, bleu2, bleu3, bleu4, losses.avg, top5accs.avg


if __name__ == '__main__':
    main()
