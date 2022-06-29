import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from random import randint
import time
import argparse

from models import Encoder, EncoderX, DecoderWithAttention, ReplayMemory
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import paths as P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
cudnn.benchmark = False  

replay_freq = 200  # memory addition

# Model and training parameters
cf = {'emb_dim': 512,
    'attention_dim': 512,
    'decoder_dim': 512,
    'dropout': 0.5,
    'start_epoch': 0,
    'epochs': 120, 
    'epochs_since_improvement': 0, 
    'batch_size': 64,
    'workers': 1, 
    'encoder_lr': 1e-4, 
    'decoder_lr': 4e-4, 
    'grad_clip': 5.,
    'alpha_c': 1., 
    'best_bleu4': 0.,
    'print_freq': 100
    }


def main(data_name, data_folder, checkpoint, enc_backbone, fine_tune_encoder, mem, target_checkpoint):
    """
    Training and validation.
    """

    global cf

    # Read word map
    word_map_file = './WORDMAP_vizwiz_5_cap_per_img_5_min_word_freq.json'
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=cf['attention_dim'],
                                       embed_dim=cf['emb_dim'],
                                       decoder_dim=cf['decoder_dim'],
                                       vocab_size=len(word_map),# vocab_size,
                                       dropout=cf['dropout'])
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=cf['decoder_lr'])
        encoder = EncoderX() if enc_backbone == 'resnext' else Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=cf['encoder_lr']) if fine_tune_encoder else None
        start_epoch = cf['start_epoch']
        epochs_since_improvement = cf['epochs_since_improvement']
        best_bleu4 = cf['best_bleu4']

    else:
        # memory path
        memory_path = checkpoint.replace('pth.tar', 'pkl').replace('BEST_', '')
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=cf['encoder_lr'])

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=cf['batch_size'], shuffle=True, num_workers=cf['workers'], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=cf['batch_size'], shuffle=True, num_workers=cf['workers'], pin_memory=True)
    
    if mem:
        if checkpoint:
            buffer = dict()
            buffer = pickle.load(open(memory_path, 'rb')) 
            memory = ReplayMemory(buffer=buffer)
        else:
            memory = ReplayMemory()

    # Epochs
    for epoch in range(start_epoch, cf['epochs']):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        if mem:
            train_with_memory(memory=memory,
                            train_loader=train_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            encoder_optimizer=encoder_optimizer,
                            decoder_optimizer=decoder_optimizer,
                            epoch=epoch)
        
        else:
            # One epoch's training
            train(train_loader=train_loader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        if mem:
            save_checkpoint(target_checkpoint, epoch, epochs_since_improvement, 
                            encoder, decoder, encoder_optimizer, decoder_optimizer, 
                            recent_bleu4, is_best, memory=memory.memory)
        else:
            save_checkpoint(target_checkpoint, epoch, epochs_since_improvement, 
                            encoder, decoder, encoder_optimizer, decoder_optimizer, 
                            recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # aliciaviernes fix
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += cf['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if cf['grad_clip'] is not None:
            clip_gradient(decoder_optimizer, cf['grad_clip'])
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, cf['grad_clip'])

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % cf['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def train_with_memory(memory, train_loader, encoder, decoder, criterion, \
    encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        writemem = randint(0, 1)
        if writemem == 1:
            # memory.push(imgs.numpy(), (imgs.numpy(), caps.numpy(), caplens.numpy()))
            # memory.push(imgs, (imgs, caps, caplens))
            memory.push(imgs, (caps, caplens))
            # print('PUSHED:', imgs.shape, caps.shape, caplens.shape)
        
        
        if (i + 1) % replay_freq == 0:

            imgsx, capsx, caplensx = memory.sample(sample_size=cf['batch_size'])

            # print('RETRIEVED:', imgsx.shape, capsx.shape, caplensx.shape)
            
            imgsx = imgsx.to(device)
            capsx = capsx.to(device)
            caplensx = caplensx.to(device)

            imgsx = encoder(imgsx)
            # print('Repeated encoded image shape:', imgsx.shape)
            scores, caps_sorted, decode_lengths, alphas, \
                sort_ind = decoder(imgsx, capsx, caplensx)
            
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, \
                batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, \
                batch_first=True).data
            
            loss = criterion(scores, targets)
            loss += cf['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()


            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if cf['grad_clip'] is not None:
                clip_gradient(decoder_optimizer, cf['grad_clip'])
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, cf['grad_clip'])

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            
        # print('imgs', imgs.shape, 'caps', caps.shape, 'caplens', caplens.shape)
        # print('types', imgs.dtype, caps.dtype, caplens.dtype)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        # print('Encoded image size:', imgs.shape)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # aliciaviernes fix
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += cf['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if cf['grad_clip'] is not None:
            clip_gradient(decoder_optimizer, cf['grad_clip'])
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, cf['grad_clip'])

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % cf['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
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

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += cf['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % cf['print_freq'] == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {101, 0}], # if w not in {word_map['<start>'], word_map['<pad>']}
                        img_caps))  # remove <start> and pads  # aliciaviernes: remove [CLS] and [PAD]
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
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Interactive Image Captioning.')
    parser.add_argument('--dataset', '-d', type=str, default='vizwiz', 
        required=False, help='Which dataset, coco or vizwiz?')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, 
        required=False, help='Path to last/best checkpoint.')
    parser.add_argument('--encoder_backbone', '-e', type=str, 
        default='resnext', help='resnet or resnext?')
    parser.add_argument('--finetune_encoder', '-f', type=bool, default=False, 
        required=False, help='Finetune encoder?')
    parser.add_argument('--input_folder', '-i', type=str, required=True, 
        help='Path to input files.')
    parser.add_argument('--memory', '-m', type=bool, default=False, 
        help='Train with memory?')
    parser.add_argument('--target_checkpoint', '-t', type=str, default=False, 
        help='Name of target checkpoint.')
    args = parser.parse_args()

    data_name = f'{args.dataset}_5_cap_per_img_5_min_word_freq'  # base name shared by data files

    # folder with data files saved by create_input_files.py
    if args.dataset == 'coco':
        data_folder = f'{P.base_coco}original_output_files'
    else:
        data_folder = args.input_folder

    main(data_name=data_name,
        data_folder=data_folder, 
        fine_tune_encoder=args.finetune_encoder, 
        enc_backbone=args.encoder_backbone, 
        mem=args.memory,
        target_checkpoint=args.target_checkpoint, 
        checkpoint=args.checkpoint)
