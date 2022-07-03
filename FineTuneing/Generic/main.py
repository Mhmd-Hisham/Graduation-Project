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

SEED = 512
MATPLOTLIB_STYLE = "seaborn"

plt.style.use(MATPLOTLIB_STYLE)

def reset_environment(reset_seed=True, seed=SEED):
    torch.cuda.empty_cache()
    gc.collect()
    if reset_seed:
        pl.seed_everything(seed)


#@title PytorchDataset
class PyTorchDataset(Dataset):
    """  PyTorch Dataset class  """
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data
        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer from hugging face transformers
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into the model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
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
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        num_workers: int = 2,
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer from hugging face
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.eval_df = eval_df

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

        self.test_dataset = PyTorchDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.eval_dataset = PyTorchDataset(
            self.eval_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

# @title LightningModel
class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        checkpoint_name: str,
        output_dir: str,
        save_only_last_epoch: bool = False,
        learning_rate: float = 0.0001,
    ):
        """
        initiates a PyTorch Lightning Model
        :param PreTrainedTokenizer tokenizer: a pre-trained tokenizer from hugging face
        :param PreTrainedModel model: a pre-trained model from hugging face (Causal)
        :param str output_dir: output directory to save model checkpoints. 
        Defaults to "outputs".
        :param bool save_only_last_epoch: If True, save just the last 
        epoch else models are saved for every epoch
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        self.average_training_loss = None
        self.average_validation_loss = None

        self.learning_rate = learning_rate
        self.checkpoint_name = checkpoint_name
        self.save_only_last_epoch = save_only_last_epoch

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
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
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
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
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

        self.log("test_loss", loss, prog_bar=True, logger=True,)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=self.learning_rate)

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )

        path = f"{self.output_dir}/{self.checkpoint_name}/epoch-{self.current_epoch}-tloss-{str(self.average_training_loss)}-vloss-{str(self.average_validation_loss)}"
        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)
        else:
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
        checkpoint_name: str,
        output_dir: str,
        learning_rate: float,
    ) -> None:
        """ initiates SimpleEncoderDecoder class """
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.checkpoint_name = checkpoint_name
        self.trainer = None
        self.callbacks = [TQDMProgressBar(refresh_rate=5)]

    def from_pretrained(
        self,
        tokenizer_class: PreTrainedTokenizer,
        model_class: PreTrainedModel,
        checkpoint_name: str,
        return_dict: bool=True,
        use_gpu: bool=True,
        fp16: bool=True) -> None:
        """
            loads a model (from a local folder or from HF) for training/fine-tuning/evaluating/inference
        """
        self.tokenizer = tokenizer_class.from_pretrained(checkpoint_name)
        self.model = model_class.from_pretrained(
            checkpoint_name, 
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
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        max_epochs: int = 5,
        use_gpu: bool = True,
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        precision=32,
        logger="default",
        dataloader_num_workers: int = 2,
        save_only_last_epoch: bool = False,
    ):
        """
        trains the model on custom a dataset
        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            test_df ([type], optional): test datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            output_dir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision (int, optional): sets precision training - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
            logger (pytorch_lightning.loggers) : any logger supported by PyTorch Lightning. Defaults to "default". If "default", pytorch lightning default logger is used.
            dataloader_num_workers (int, optional): number of workers in train/test/val dataloader
            save_only_last_epoch (bool, optional): If True, saves only the last epoch else models are saved at every epoch
        """
        self.data_module = LightningDataModule(
            train_df,
            test_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        self.lightning_model = LightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            checkpoint_name=self.checkpoint_name,
            output_dir=self.output_dir,
            save_only_last_epoch=save_only_last_epoch,
        )

        # add callbacks for early stopping
        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.01,
                patience=early_stopping_patience_epochs,
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
            max_epochs=max_epochs,
            gpus=gpus,
            precision=precision,
            log_every_n_steps=1,
        )

        # fit trainer
        self.trainer.fit(self.lightning_model, self.data_module)

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates predictions
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds

    def predict_multiple(
        self,
        source_text: List[str],
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        ) -> List[str]:
        input_ids = self.tokenizer(
            source_text, 
            return_tensors="pt",
            add_special_tokens=True,
            padding='max_length',
            max_length=265,
            truncation=True
        )['input_ids']

        input_ids = input_ids.to(self.device)

        generated_samples = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )

        return [self.tokenizer.decode(
                g, 
                skip_special_tokens=skip_special_tokens, 
                clean_up_tokenization_spaces=clean_up_tokenization_spaces) 
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

