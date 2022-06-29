import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import paths as P
import config as C

# Parameters -- aliciaviernes modification
dss = ['coco_', 'vizwiz_']
base = '5_cap_per_img_5_min_word_freq' # base name shared by data files
w_base = 'bert' # aliciaviernes: training suffix

# NOTE CHECKPOINT:  COCO_BERT
# NOTE TEST DATA:   COCO
checkpoint = f'./checkpoints_decoder_encoder/BEST_checkpoint_{dss[0]}{w_base}.pth.tar'  # model checkpoint
# checkpoint = f'BEST_checkpoint_{dss[0]}{w_base}_plus_encoder.pth.tar'

# Folder for TEST data [change both data_name and data_folder]
data_name = f'{dss[1]}{base}'

# folder with data files saved by create_input_files.py
if data_name == f'{dss[0]}{base}':
    data_folder = f'{P.base_coco}original_output_files/' 
elif data_name == f'{dss[1]}{base}':
    data_folder = f'{P.base_vizwiz}output_files_cocoplus'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()


vocab_size = C.vocab_size

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size=1):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        # k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
        k_prev_words = torch.LongTensor([[101]] * k).to(device)  # NOTE

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        # print('Seqs:', seqs)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)()

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1) # (s, step+1)
            # print('Seqs 2:', seqs)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != 102]  # != word_map['<end>']
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            # print('Complete inds:', complete_inds)

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds.long()[incomplete_inds]]
            c = c[prev_word_inds.long()[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds.long()[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        # print('Complete seqs scores:', complete_seqs_scores)

        if complete_seqs_scores != list():
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {101, 102, 0}],
                    img_caps))  # remove <start> and pads
                    # word_map['<start>'], word_map['<end>'], word_map['<pad>']
            references.append(img_captions)
            # print(img_captions)

            # Hypotheses
            hypotheses.append([w for w in seq if w not in {101, 102, 0}])
            # print([w for w in seq if w not in {101, 102, 0}]) 
            # word_map['<start>'], word_map['<end>'], word_map['<pad>']

            assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    # print('Refs:', references)
    # print('Hyps:', hypotheses)
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':

    beam_size = 1
    # print(checkpoint + '\n')
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
