
import json

from tqdm import tqdm
from transformers import BertTokenizer
import jieba
tokenizer = BertTokenizer.from_pretrained("bart-base-chinese-cluecorpussmall")

def load_jsonl(path):
    print(f"Read text file from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.writelines(json.dumps(line, ensure_ascii=False, indent=4) + "\n")
    print(f"Data has been saved to {path}.")

from glob import glob

def convert_bio_label(bio_indexes, total_len):
    labels = [0] * total_len
    for (start, end) in bio_indexes:
        labels[start] = 1
        for i in range(start + 1, end + 1):
            labels[i] = 2
    return labels

files = glob("data_mc/*.jsonl")
for file in files:
    data = load_jsonl(file)
    json_data = []
    for d in tqdm(data):
        src = []
        for utterance in d['content']:
            tokens = list(jieba.cut(utterance['utterance'], cut_all=False))
            utterance['utterance'] = ' '.join(tokens)
        if len(src) < len(d['labels']):
            d['labels'] = d['labels'][:len(src)]
            # print(d['id'] + '  ' + str(len(src)) + '  ' + str(len(d['labels'])))
        elif len(src) > len(d['labels']):
            d['labels'].extend(['0'] * (len(src) - len(d['labels'])))
        dialogue = d['content']

        json_data.append({
            "Dialogue": dialogue,
            "UserSumm": d['summary']['description'],
            "AgentSumm": d['summary']['suggestion'],
            "FinalSumm": d['summary']['description'] + d['summary']['suggestion']
        })
    data = []
    data.append(json_data)
    file = file.split('.')[0] + '.json'
    save_jsonl(data, f"{ file }")
