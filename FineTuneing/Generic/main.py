#@title Imports

import gc
import os
import time
import functools
import multiprocessing
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelWithLMHead,
    AutoTokenizer,
)
import evaluate 

import glob
import json

SEED = 512
MATPLOTLIB_STYLE = "seaborn"

plt.style.use(MATPLOTLIB_STYLE)

def reset_environment(reset_seed=True, seed=SEED):
    torch.cuda.empty_cache()
    gc.collect()
    if reset_seed:
        pl.seed_everything(seed)

#@title LightningDataset
class LightningDataset(Dataset):
    """  PyTorch Dataset class  """
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        params: ExperimentParameters
        ):
        """
        initiates a PyTorch Dataset Module for input data
        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer from hugging face transformers
        """
        self.tokenizer = tokenizer
        self.data = data
        self.params = params

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into the model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,

            return_attention_mask=True,
            return_tensors="pt",

            max_length=self.params['encoder']['source_max_token_len'],
            padding=self.params['encoder']['padding'],
            truncation=self.params['encoder']['truncation'],
            add_special_tokens=self.params['encoder']['add_special_tokens'],
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],

            max_length=self.params['encoder']['target_max_token_len'],
            padding=self.params['encoder']['padding'],
            truncation=self.params['encoder']['truncation'],
            add_special_tokens=self.params['encoder']['add_special_tokens'],

            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        # to make sure we have correct labels for the text generation
        labels[labels == 0] = -100

        return {
            "source_text_input_ids": source_text_encoding["input_ids"].flatten(),
            "source_text_attention_mask": source_text_encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
            "labels_attention_mask": target_text_encoding["attention_mask"].flatten(),
        }

#@title LightningDataModule
class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        params: ExperimentParameters,
        ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            val_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): test dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer from hugging face
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.eval_df = eval_df

        self.tokenizer = tokenizer
        self.params = params

    def setup(self, stage=None):
        self.train_dataset = LightningDataset(
            self.train_df,
            self.tokenizer,
        )

        self.test_dataset = LightningDataset(
            self.test_df,
            self.tokenizer,
        )

        self.eval_dataset = LightningDataset(
            self.eval_df,
            self.tokenizer,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.params['dataset_loader']['batch_size'],
            num_workers=self.params['dataset_loader']['num_workers'],
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.params['dataset_loader']['batch_size'],
            num_workers=self.params['dataset_loader']['num_workers'],
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.eval_dataset,
            shuffle=False,
            batch_size=self.params['dataset_loader']['batch_size'],
            num_workers=self.params['dataset_loader']['num_workers'],
        )

# @title LightningModel
class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        params: ExperimentParameters,
    ):
        """
        initiates a PyTorch Lightning Model
        :param PreTrainedTokenizer tokenizer: a pre-trained tokenizer from hugging face
        :param PreTrainedModel model: a pre-trained model from hugging face (Causal)
        Defaults to "outputs".
        epoch else models are saved for every epoch
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.params = params

        self.average_training_loss = None
        self.average_validation_loss = None

        self.checkpoint_name = self.params['model']['checkpoint_name']

        self.checkpoints_dir = f"{self.params['general']['output_dir']}/{self.checkpoint_name}/"

        self.experiment_version = list(
            sorted(map(lambda s: int(s[s.rfind('version')+8:]),
                   ['version_-1']+glob.glob(self.checkpoints_dir+"/version_*")),
                   reverse=True)
            )[0]+1

        self.checkpoints_dir += f"version_{self.experiment_version}/"

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "Loss/train", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "Loss/validation", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("Loss/test", loss, prog_bar=True, logger=True,)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=self.params['model']['learning_rate'])

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )

        path = self.checkpoints_dir \
              +f"-epoch-{self.current_epoch}" \
              +f"-tloss-{str(self.average_training_loss)}" \
              +f"-vloss-{str(self.average_validation_loss)}"

        if self.params['model']['save_every_epoch']:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)
        else:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )

#@title SimpleEncoderDecoder class
class SimpleEncoderDecoder:
    """ A Generic class for fine-tuning Encoder-Decoder architectures (loads the models the transformers library)"""

    def __init__(self,
        params: ExperimentParameters,
    ) -> None:
        """ initiates SimpleEncoderDecoder class """
        self.trainer = None
        self.callbacks = [TQDMProgressBar(refresh_rate=5)]
        self.params = params
        self.checkpoint_name = self.params['model']['checkpoint_name']

    def from_pretrained(
        self,
        tokenizer_class: PreTrainedTokenizer,
        model_class: PreTrainedModel,
        return_dict: bool=True,
        use_gpu: bool=True,
        fp16: bool=True) -> None:
        """
            loads a model (from a local folder or from HF) for training/fine-tuning/evaluating/inference
        """
        self.tokenizer = tokenizer_class.from_pretrained(self.checkpoint_name)
        self.model = model_class.from_pretrained(
            self.checkpoint_name, 
            return_dict=return_dict # set this to false during inference
        )

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        if fp16:
            self.model.half()

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        logger="default",
        use_gpu: bool = True,
    ):
        """
        trains the model on custom a dataset
        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            test_df ([type], optional): test datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"

            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            logger (pytorch_lightning.loggers) : any logger supported by PyTorch Lightning. Defaults to "default". If "default", pytorch lightning default logger is used.
        """
        self.data_module = LightningDataModule(
            train_df,
            test_df,
            eval_df,
            self.tokenizer,
            batch_size=self.params['trainer']['batch_size'],
            source_max_token_len=self.params['trainer']['source_max_token_len'],
            target_max_token_len=self.params['trainer']['target_max_token_len'],
            num_workers=self.params['dataset_loader']['num_workers'],
        )

        self.lightning_model = LightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            checkpoint_name=self.checkpoint_name,
        )

        # add callbacks for early stopping
        if self.params['trainer']['early_stopping_patience_epochs'] > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=self.params['trainer']['early_stopping_min_delta'],
                patience=self.params['trainer']['early_stopping_patience_epochs'],
                verbose=True,
                mode="min",
            )
            self.callbacks.append(early_stop_callback)

        # add gpu support
        gpus = 1 if use_gpu else 0

        # add logger
        loggers = True if logger == "default" else logger

        # prepare trainer
        self.trainer = pl.Trainer(
            logger=loggers,
            callbacks=self.callbacks,
            max_epochs=self.params['trainer']['max_epochs'],
            precision=self.params['trainer']['precision'],
            gpus=gpus,
            log_every_n_steps=1,
        )

        # fit trainer
        self.trainer.fit(self.lightning_model, self.data_module)

    def predict(self, source_text: str) -> List[str]:
        """
        generates predictions
        Args:
            source_text (str): any text for generating predictions
        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )

        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=self.params['generator']['num_beams'],
            max_length=self.params['generator']['max_length'],
            repetition_penalty=self.params['generator']['repetition_penalty'],
            length_penalty=self.params['generator']['length_penalty'],
            early_stopping=self.params['generator']['early_stopping'],
            top_p=self.params['generator']['top_p'],
            top_k=self.params['generator']['top_k'],
            num_return_sequences=self.params['generator']['num_return_sequences'],
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=self.params['generator']['skip_special_tokens'],
                clean_up_tokenization_spaces=self.params['generator']['clean_up_tokenization_spaces'],
            )
            for g in generated_ids
        ]
        return preds

    def predict_multiple(self,source_text: List[str]) -> List[str]:
        input_ids = self.tokenizer(
            source_text, 
            return_tensors="pt",

            padding=self.params['encoder']['padding'],
            truncation=self.params['encoder']['truncation'],
            add_special_tokens=self.params['encoder']['add_special_tokens'],
        )['input_ids']

        input_ids = input_ids.to(self.device)

        generated_samples = self.model.generate(
            input_ids=input_ids,
            num_beams=self.params['generator']['num_beams'],
            max_length=self.params['generator']['max_length'],
            repetition_penalty=self.params['generator']['repetition_penalty'],
            length_penalty=self.params['generator']['length_penalty'],
            early_stopping=self.params['generator']['early_stopping'],
            top_p=self.params['generator']['top_p'],
            top_k=self.params['generator']['top_k'],
            num_return_sequences=self.params['generator']['num_return_sequences'],
        )

        return [self.tokenizer.decode(
                g, 
                skip_special_tokens=self.params['generator']['skip_special_tokens'], 
                clean_up_tokenization_spaces=self.params['generator']['clean_up_tokenization_spaces']) 
            for g in generated_samples
        ]

    def batch_predict(
        self, 
        batch_size: int, 
        sequences: List[str], 
        **kwargs
        ) -> List[str]:
        """
            Computes the predictions on batches and then concatenates.
        """
        output = []
        n_steps = (len(sequences)//batch_size)
        for i in range(0, n_steps*batch_size, batch_size):
            output += self.predict_multiple(
                sequences[i:i+batch_size], 
                **kwargs)
        output += self.predict_multiple(
          sequences[n_steps*batch_size:], 
          **kwargs
        )
        return output


class ExperimentParameters(dict):
    def __init__(self):
        super().__init__(self)

    def __str__(self):
        return json.dumps(self)

    def to_json(self, filepath):
        with open(filepath, "w+") as fh:
            fh.write(json.dump(self))

parameters = ExperimentParameters()
parameters['trainer'] = {
    "batch_size": 8,
    "max_epochs": 5,
    "precision": 32,
    "early_stopping_patience_epochs": 0,  # 0 to disable early stopping feature
    "early_stopping_min_delta": 0.01,
}

parameters['general'] = {
    "source_max_token_len": 512,
    "target_max_token_len": 512,
    "output_dir":"",
    'seed':SEED
}

parameters['encoder'] = {
    "padding":"max_length",
    "truncation":True,
    "add_special_tokens": True,
}

parameters['dataset_loader'] = {
    "batch_size": 4,
    "num_workers": 2,
}

parameters['model'] = {
    "checkpoint_name":"",
    "save_every_epoch": True,
    "learning_rate": 1e-4,
}

parameters['generator'] = {
    "num_beams": 2,
    "max_length": 512,
    "repetition_penalty": 2.5,
    "length_penalty": 1.0,
    "early_stopping": True,
    "top_p": 0.95,
    "top_k": 50,
    "num_return_sequences": 1,
    "skip_special_tokens": True,
    "clean_up_tokenization_spaces": True,
}
