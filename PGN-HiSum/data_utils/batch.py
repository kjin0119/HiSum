import json
import torch
import nltk
import re
import jieba

from data_utils.vocab_gen import Vocab, article2ids, abstract2ids
from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler
from utils.config import args
from utils import config
from data_utils.tokenizer import Tokenizer, get_dialogue_vocab, load_vocab, get_tencent_embedding
from pathlib import Path
from collections import Counter
import math


dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '<PAD>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '<UNK>' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<START>' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<END>' # This has a vocab id, which is used at the end of untruncated target sequences



class PGNDataset(Dataset):
    def __init__(self, features):
        self.data = features
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        return data

def get_data(data_pth):
    with open(data_pth, 'r') as f:
        data = json.load(f)
    return data


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."

def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target

def get_border_id(sample):
    borders = [[], []]
    for topic in sample['Topics']:
        for tri in topic['Triplets']:
            min_index = min(tri['QueSummUttIDs'] + tri['AnsSummUttIDs'])
            max_index = max(tri['QueSummUttIDs'] + tri['AnsSummUttIDs'])
            borders[0].append(min_index)
            borders[1].append(max_index)
    return borders

def calculate_tf_isf(word_list):
    
    tf_scores = []
    for sentence in word_list:
        tf_score = Counter(sentence)
        sentence_length = len(sentence)
        for word in tf_score:
            tf_score[word] = tf_score[word] / sentence_length
        tf_scores.append(tf_score)
    sf_scores = Counter([word for sentence in word_list for word in sentence])
    total_sentences = len(word_list)
    isf_scores = {}
    for word in sf_scores:
        isf_scores[word] = math.log(total_sentences / sf_scores[word])
    tf_isf_scores = []
    for sentence in tf_scores:
        sentence_score = {}
        for word in sentence:
            sentence_score[word] = sentence[word] * isf_scores.get(word, 0)
        tf_isf_scores.append(sentence_score)
    return tf_isf_scores

def convert_example_to_feature(data, max_seq_length, org_vocab, sum_mode, context_mode, get_vocab=False):
    features = []
    num_len_below, num_len_over = 0, 0
    sums, contexts, contexts_joint = [], [], []
    tf_isfs, sentence_masks = [], []
    for sample in data:
        #border_ids = get_border_id(sample)
        if sum_mode == 'final':
            summary = sample['FinalSumm'] # list
        elif sum_mode == 'user':
            summary = sample['UserSumm']
        elif sum_mode == 'agent':
            summary = sample['AgentSumm']
        summary = ''.join(summary) # str, no split
        sums.append(summary)
        context = []
        sentence_array = []
        for turn in sample['Dialogue']:
            tmp_utt = []
            # if turn['turn'] in border_ids[0]:
            #     tmp_utt.append('<BOP>')
            if turn['speaker'] == 'Q':
                tmp_utt += [sample['QRole'], ':']
            else:
                tmp_utt += ['客服', ':']
            sent = turn['utterance'].split()
            for word in sent:
                if len(word) > 2 and word[0] == '[' and word[-1] == ']':
                    tmp_utt += ['[', word[1:-1], ']']
                else:
                    tmp_utt.append(word)
            # if turn['turn'] in border_ids[1]:
            #     tmp_utt.append('<EOP>')
            tmp_utt = ' '.join(tmp_utt)
            sen = tmp_utt.replace(" ", "").strip()
            sentence_array.append(sen)
            if context_mode == 'both':
                context.append(tmp_utt)
            elif context_mode == 'user' and turn['speaker'] == 'Q':
                context.append(tmp_utt)
            elif context_mode == 'agent' and turn['speaker'] == 'A':
                context.append(tmp_utt)
            from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bart-base-chinese-cluecorpussmall")
        sentence_mask = []
        for i in range(len(sentence_array)):
            if i == 0:
                sentence_mask.extend([2] * 6 + [i + 3] * (len(tokenizer.tokenize(sentence_array[i])) - 6))
            else:
                sentence_mask.extend([i + 3] * len(tokenizer.tokenize(sentence_array[i])))

        tmp = [element[3:] for element in sentence_array if element]
        tf_isf = calculate_tf_isf(tmp)
        average_tf_isf_scores = [
            sum([sentence_score.get(word, 0) for word in jieba.lcut(seg)]) / len(jieba.lcut(seg))
            for seg, sentence_score in zip(tmp, tf_isf)]
        tf_isf_score = [list(d.values()) for d in tf_isf]
        average_tf_isf_scores = [score if score >= 0.001 else 0.001 for score in average_tf_isf_scores]
        for i in range(len(tf_isf_score)):
            if len(tf_isf_score[i]) <= 6:
                average_tf_isf_scores[i] = 0.001
        sentence_masks.append(sentence_mask)
        tf_isfs.append(average_tf_isf_scores)
        context_joint = " <EOU> ".join(context)
        contexts.append(context)
        contexts_joint.append(context_joint)


    # get vocab
    if get_vocab:
        if args.new_vocab:
            get_dialogue_vocab(sums, contexts, Path('data_utils/embeddings/'), 'word')
            dialogue_vocab = load_vocab(Path('data_utils/embeddings/dialogue_vocab_word'))
            get_tencent_embedding(dialogue_vocab, Path('data_utils/embeddings/Tencent_AILab_ChineseEmbedding.txt'),
                                  Path(args.vocab_path))
        vocab = Tokenizer(args.vocab_path, args.vocab_dim, args.vocab_size)
    else:
        vocab = org_vocab

    start_decoding = vocab.token2idx(START_DECODING)
    stop_decoding = vocab.token2idx(STOP_DECODING)

    for summary, context, context_joint, sentence_mask, tf_isf in zip(sums, contexts, contexts_joint, sentence_masks, tf_isfs):
        if args.split_word:
            sum_ids = vocab.tokenize(summary, False)
            context_ids = vocab.tokenize(context_joint, True)
        else:
            summary = ' '.join(list(summary))
            sum_ids = vocab.tokenize(summary, True)
            context = [' '.join(list(''.join(s.split()))) for s in context]
            context_joint = ' <EOU> '.join(context)
            context_ids = vocab.tokenize(context_joint, True)
        if len(context_ids) > max_seq_length:
            num_len_over += 1
        else:
            num_len_below += 1
        context_ids = context_ids[:max_seq_length]

        dec_input, dec_output = get_dec_inp_targ_seqs(sum_ids, args.max_dec_steps, start_decoding, stop_decoding)
        enc_input_extend, article_oovs = vocab.article2ids(context_joint.split()[:max_seq_length])
        if args.split_word:
            dec_extend = vocab.abstract2ids(vocab.split(summary), article_oovs)
        else:
            dec_extend = vocab.abstract2ids(summary.split(), article_oovs)
        dec_input_extend, dec_output_extend = get_dec_inp_targ_seqs(dec_extend, args.max_dec_steps, start_decoding, stop_decoding)


        features.append(
            {'enc_input': context_ids,
             'dec_input': dec_input,
             'dec_output': dec_output,
             'enc_input_extend': enc_input_extend,
             'dec_input_extend': dec_input_extend,
             'dec_output_extend': dec_output_extend,
             'oovs': article_oovs,
             'pad': vocab.token2idx(PAD_TOKEN),
             'summ_sent': summary,
             'sentence_mask': sentence_mask,
             'tf_isf': tf_isf}
            )
    print(num_len_below, num_len_over)

    return features, vocab

def padded_sequence(seqs, pad):
    max_len = max([len(seq) for seq in seqs])
    padded_seqs = [seq + [pad] * (max_len - len(seq)) for seq in seqs]
    length = [len(seq) for seq in seqs]
    return padded_seqs, length

def batchify_data(batch):
    enc_input_ids = [f['enc_input'] for f in batch]
    enc_input_ids, enc_lens = padded_sequence(enc_input_ids, batch[0]['pad'])
    enc_input_ids = torch.tensor(enc_input_ids, dtype=torch.long)
    enc_lens = torch.tensor(enc_lens, dtype=torch.long)
    enc_batch_extend_vocab = [f['enc_input_extend'] for f in batch]
    enc_batch_extend_vocab, _ = padded_sequence(enc_batch_extend_vocab, batch[0]['pad'])
    enc_batch_extend_vocab = torch.tensor(enc_batch_extend_vocab, dtype=torch.long)
    art_oovs = [f['oovs'] for f in batch]
    max_art_oovs = max([len(f['oovs']) for f in batch])

    dec_input_ids = [f['dec_input'] for f in batch]
    dec_output_ids = [f['dec_output'] for f in batch]
    if config.pointer_gen:
        dec_output_ids = [f['dec_output_extend'] for f in batch]
    dec_input_ids, dec_lens = padded_sequence(dec_input_ids, batch[0]['pad'])
    dec_output_ids, dec_lens = padded_sequence(dec_output_ids, batch[0]['pad'])
    dec_input_ids = torch.tensor(dec_input_ids, dtype=torch.long)
    dec_output_ids = torch.tensor(dec_output_ids, dtype=torch.long)
    dec_lens = torch.tensor(dec_lens, dtype=torch.long)
    summ_sent = [f['summ_sent'] for f in batch]
    sentence_mask = [f['sentence_mask'] for f in batch]
    tf_isf = [f['tf_isf'] for f in batch]
    return ((enc_input_ids, enc_lens, enc_batch_extend_vocab, art_oovs, max_art_oovs, sentence_mask, tf_isf), (dec_input_ids, dec_output_ids, dec_lens, summ_sent))

def get_train_dataloader(data_pth, max_seq_length, train_batch_size):
    print('processing training data-------------')
    data = get_data(data_pth)
    features, vocab = convert_example_to_feature(data, max_seq_length, None, sum_mode=args.sum_mode, context_mode=args.context_mode, get_vocab=True)
    train_data = PGNDataset(features)
    sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=train_batch_size, collate_fn=batchify_data)
    return train_dataloader, features, vocab

def get_val_dataloader(data_pth, vocab, max_seq_length, val_batch_size, mode='eval', beam_size=0):
    data = get_data(data_pth)
    features, vocab = convert_example_to_feature(data, max_seq_length, vocab, sum_mode=args.sum_mode, context_mode=args.context_mode, get_vocab=False)
    if mode == 'eval':
        val_data = PGNDataset(features)
        sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=val_batch_size, collate_fn=batchify_data)
    else:
        decode_features = []
        for f in features:
            decode_features.extend([f] * beam_size)
        val_data = PGNDataset(decode_features)
        sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=beam_size, collate_fn=batchify_data)
    return val_dataloader, features




