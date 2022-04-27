import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from dataprovider.DataProvider import DatasetProvider

class T5Model(pl.LightningModule):

    def __init__(self, hparams):
        super(T5Model, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)

        self.data_dir=hparams.data_dir
        self.output_dir=hparams.output_dir
        self.model_name_or_path=hparams.model_name_or_path
        self.tokenizer_name_or_path=hparams.tokenizer_name_or_path
        self.train_dataset=hparams.train_dataset
        self.test_dataset=hparams.test_dataset
        self.max_seq_length=hparams.max_seq_length
        self.learning_rate=hparams.learning_rate
        self.weight_decay=hparams.weight_decay
        self.adam_epsilon=hparams.adam_epsilon
        self.warmup_steps=hparams.warmup_steps
        self.train_batch_size=hparams.train_batch_size
        self.eval_batch_size=hparams.eval_batch_size
        self.num_train_epochs=hparams.num_train_epochs
        self.gradient_accumulation_steps=hparams.gradient_accumulation_steps
        self.n_gpu=hparams.n_gpu
        self.early_stop_callback=hparams.early_stop_callback
        self.fp_16=hparams.fp_16
        self.opt_level=hparams.opt_level
        self.max_grad_norm=hparams.max_grad_norm
        self.seed=hparams.seed



    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask, 
            labels=lm_labels
        )

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def _step(self, batch):
        lm_labels = batch["tid"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["sid"], 
            attention_mask=batch["smask"],
            lm_labels=lm_labels, 
            decoder_attention_mask=batch['tmask']
        )
        return outputs[0] #loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {
            "avg_train_loss": avg_train_loss, 
            "log": tensorboard_logs, 
            'progress_bar': tensorboard_logs
        }

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
            "avg_val_loss": avg_loss, 
            "log": tensorboard_logs, 
            'progress_bar': tensorboard_logs
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.learning_rate, 
            eps=self.adam_epsilon
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, using_native_amp=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = DatasetProvider(
            tokenizer=self.tokenizer, 
            type_path=self.train_dataset, 
            data_dir=self.data_dir, 
            max_len=self.max_seq_length
        )
        dataloader = DataLoader(
            train_dataset, 
            batch_size=self.train_batch_size, 
            drop_last=True, 
            shuffle=True, 
            num_workers=4
        )
        t_total = (
                (len(dataloader.dataset) // (self.train_batch_size * max(1, self.n_gpu)))
                // self.gradient_accumulation_steps
                * float(self.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = DatasetProvider(
            tokenizer=self.tokenizer, 
            type_path=self.test_dataset, 
            data_dir=self.data_dir, 
            max_len=self.max_seq_length
        )
        return DataLoader(
            val_dataset, 
            batch_size=self.eval_batch_size, 
            num_workers=4
        )