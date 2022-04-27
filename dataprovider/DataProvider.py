from lib2to3.pgen2 import token
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import csv
import os

class DatasetProvider(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=256):
        self.path = os.path.join(data_dir, type_path + '.tsv')
        print(self.path)

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.max_len = max_len
        self.data = []
        self.inputs = []
        self.targets = []

        with open(self.path, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                self.data.append(row)

        for element in self.data:
            input = "paraphrase: " + element[0] + ' </s>'
            target = element[1] + " </s>"
            t_input = self.tokenizer.batch_encode_plus([input], max_length=self.max_len, pad_to_max_length=True, truncation=True, return_tensors="pt")
            t_target = self.tokenizer.batch_encode_plus([target], max_length=self.max_len, pad_to_max_length=True, truncation=True, return_tensors="pt")
            self.inputs.append(t_input)
            self.targets.append(t_target)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sid = self.inputs[index]["input_ids"].squeeze()
        tid = self.targets[index]["input_ids"].squeeze()
        smask = self.inputs[index]["attention_mask"].squeeze()
        tmask = self.targets[index]["attention_mask"].squeeze()
        return {"source_ids": sid, "source_mask": smask, "target_ids": tid, "target_mask": tmask}        