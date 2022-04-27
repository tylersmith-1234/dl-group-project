from torch.utils.data import Dataset
from transformers import T5Tokenizer
import csv
import os

tokenizer = T5Tokenizer.from_pretrained('t5-base')

class MSRPCDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=256):
        self.path = os.path.join(data_dir, type_path + '.csv')

        self.source_column = "question1"
        self.target_column = "question2"

        self.max_len = max_len
        self.tokenizer = tokenizer

        self.data = []

        self.inputs = []
        self.targets = []

        with open(self.path, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.data.append(row)

        # self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sid = self.inputs[index]["input_ids"].squeeze()
        tid = self.targets[index]["input_ids"].squeeze()
        smask = self.inputs[index]["attention_mask"].squeeze()
        tmask = self.targets[index]["attention_mask"].squeeze()
        return {"sid": sid, "smask": smask, "tid": tid, "tmask": tmask}

    def _build(self):
        for example in self.data:
            input = example[0]
            target = example[1]
            input = "paraphrase: " + input + ' </s>'
            target = target + " </s>"
            t_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, pad_to_max_length=True, truncation=True, return_tensors="pt"
            )
            t_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, pad_to_max_length=True, truncation=True, return_tensors="pt"
            )
            self.inputs.append(t_input)
            self.targets.append(t_target)