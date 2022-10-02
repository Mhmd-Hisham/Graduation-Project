
#@markdown Importing modules

import os
import glob
import shutil
import multiprocessing
from typing import List, Tuple, Dict, Callable

# to show the full dialogues in the dataframes
import pandas as pd

import torch
import numpy as np

from transformers import (
    T5ForConditionalGeneration,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
)

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar


# the parameters passed to the tokenizer when encoding text
encoding_parameters = {
    "padding": "max_length",
    "truncation": True,
    "max_length": 512,
    "add_special_tokens": True,
}

# the parameters passed to the model when decoding text
decoding_parameters = {
}

other_parameters = {
    # the batch size at which the data is loaded into memory
    "data_module_batch_size": 4,

    # the number of cpu cores in the current machine
    "cpu_cores": multiprocessing.cpu_count(),
    
    # saves the last recent 'n' epochs
    "save_last_n_epochs": 3,

}

optimizer_parameters = {
    # the fixed learning rate for the model
    "fixed_learning_rate": 0.0001,
}

arguments = {
    "encoding": encoding_parameters,
    "decoding": decoding_parameters,
    "optimizer": optimizer_parameters,
    "other": other_parameters,
}

#@title LightningDataset
class LightningDataset(Dataset):
    """  PyTorch Dataset class  """
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: Dict[str, Dict],
    ):
        """
        Initiates a PyTorch Dataset Module for input data

        Parameters
        ----------
            data : input pandas dataframe. Dataframe must have 2 column: "source_text" and "target_text"
            tokenizer : a PreTrainedTokenizer from hugging face.
            args : a nested dictionary that contains all the parameters for encoding/decoding 
        """
        self.data = data
        self.tokenizer = tokenizer
        self.encoding_args = args['encoding']

    def __len__(self) -> int:
        """ returns the length of data """
        return self.data.shape[0]

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5 model"""

        row = self.data.iloc[index]

        encoded_source = self.tokenizer(
            row.source_text,
            max_length=self.encoding_args['max_length'],
            padding=self.encoding_args['padding'],
            truncation=self.encoding_args['truncation'],
            add_special_tokens=self.encoding_args['add_special_tokens'],

            return_attention_mask=True,
            return_tensors="pt",
        )

        encoded_target = self.tokenizer(
            row.target_text,
            max_length=self.encoding_args['max_length'],
            padding=self.encoding_args['padding'],
            truncation=self.encoding_args['truncation'],
            add_special_tokens=self.encoding_args['add_special_tokens'],

            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = encoded_target["input_ids"]

        # to make sure we have correct labels for T5 text generation
        labels[labels == 0] = -100

        return {
            "source_text_input_ids": encoded_source["input_ids"].flatten(),
            "source_text_attention_mask": encoded_source["attention_mask"].flatten(),
            "labels": labels.flatten(),
            "labels_attention_mask": encoded_target["attention_mask"].flatten(),
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
        args: Dict[str, Dict],
    ):
        """
        initiates a PyTorch Lightning Data Module

        Parameters
        ----------
            train_df : training dataframe. Dataframe must contain 2 columns: "source_text" and "target_text"
            test_df : validation dataframe. Dataframe must contain 2 columns: "source_text" and "target_text"
            tokenizer: a PreTrainedTokenizer object from hugging face
            batch_size: batch size of loading the data.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.eval_df = eval_df
        self.tokenizer = tokenizer
        self.args = args

        self.num_workers = arguments['other']['cpu_cores']
        self.batch_size = arguments['other']['data_module_batch_size']

    def setup(self, stage=None):
        self.train_dataset = LightningDataset(self.train_df, self.tokenizer, self.args)
        self.test_dataset = LightningDataset(self.test_df, self.tokenizer, self.args)
        self.val_dataset = LightningDataset(self.eval_df, self.tokenizer, self.args)

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
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

#@title LightningModel
class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(
        self,
        tokenizer,
        model,
        checkpoint_name: str,
        checkpoints_dir: str,
        args: Dict[str, Dict],
    ):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            checkpoints_dir : output directory to save model checkpoints.
        """
        super().__init__()
        self.model = model
        self.checkpoint_name = checkpoint_name
        self.tokenizer = tokenizer
        self.args = args
        self.save_last_n_epochs = args['other']['save_last_n_epochs']
        self.saved_epochs = []

        self.average_training_loss = None
        self.average_validation_loss = None

        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir = os.path.join(checkpoints_dir, checkpoint_name)

        # gets the next experiment version
        self.experiment_version = list(
            sorted(
                map(lambda s: int(s[s.rfind('version')+8:]),
                   ['version_-1']+glob.glob(self.checkpoints_dir+"/version_*")),
                   reverse=True)
            )[0]+1

        self.checkpoints_dir = os.path.join(self.checkpoints_dir, f"version_{self.experiment_version}/")

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
        # return Adafactor(
        #     self.parameters(),
        #     lr=1e-3,
        #     eps=(1e-30, 1e-3),
        #     clip_threshold=1.0,
        #     decay_rate=-0.8,
        #     beta1=None,
        #     weight_decay=0.0,
        #     relative_step=False,
        #     scale_parameter=False,
        #     warmup_init=False
        # )

        # this is the old optimizer
        return AdamW(
            self.parameters(), 
            lr=self.args['optimizer']['fixed_learning_rate']
        )

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )

        if self.save_last_n_epochs == 0:
            return

        # get the name of the least recent epoch
        to_remove = ""
        if len(self.saved_epochs) == self.save_last_n_epochs:
            to_remove = self.saved_epochs.pop(0)

        epoch_name = "epoch-{:03}-tloss-{:.04f}-vloss-{:.04f}"
        epoch_name = epoch_name.format(
            self.current_epoch,
            self.average_training_loss,
            self.average_validation_loss
        )
        path = os.path.join(self.checkpoints_dir, epoch_name)

        # save the current epoch
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        self.saved_epochs.append(path)

        # remove the least recent epoch (if needed)
        # only remove after saving
        if to_remove:
            shutil.rmtree(to_remove)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )

#@title EasyT5 class
class EasyT5:
    """ Custom class for fine-tunning/training T5 models"""

    def __init__(self, checkpoint_name: str, checkpoints_dir: str, args: Dict[str, Dict]):
        """ Initiates the EasyT5 class """
        self.checkpoints_dir = checkpoints_dir
        self.checkpoint_name = checkpoint_name
        self.trainer = None
        self.callbacks = [TQDMProgressBar(refresh_rate=5)]
        self.args = args

    def from_pretrained(self, model_name: str) -> None:
        """
            Loads T5 Model model for fine-tunning/training

            Parameters
            ----------
                model_name: exact model architecture name. Ex: "t5-base" or "t5-large".
        """
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{model_name}",
            return_dict=True
        )

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        max_epochs: int = 5,
        use_gpu: bool = True,
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        precision=32,
        logger="default",
    ):
        """
            trains T5/MT5 model on custom dataset
            Parameters
            ----------
                train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
                test_df ([type], optional): test datarame. Dataframe must have 2 column --> "source_text" and "target_text"
                eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"
                max_epochs (int, optional): max number of epochs. Defaults to 5.
                use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
                early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
                precision (int, optional): sets precision training - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
                logger (pytorch_lightning.loggers) : any logger supported by PyTorch Lightning. Defaults to "default". If "default", pytorch lightning default logger is used.
        """
        self.data_module = LightningDataModule(
            train_df,
            test_df,
            eval_df,
            self.tokenizer,
            self.args
        )

        self.T5Model = LightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            checkpoint_name=self.checkpoint_name,
            checkpoints_dir=self.checkpoints_dir,
            args=self.args,
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
        self.trainer.fit(self.T5Model, self.data_module)

    def load_model(
        self,
        model_dir: str = "outputs",
        use_gpu: bool = False,
        ):
        """
        loads a checkpoint for inferencing/prediction
        Args:
            model_dir (str, optional): path to model directory. Defaults to "outputs".
            use_gpu (bool, optional): if True, model uses gpu for inferencing/prediction. Defaults to True.
        """
        self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model
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
        max_length: int = 256,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        input_ids = self.tokenizer(
            source_text, 
            return_tensors="pt",
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
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

    def batch_predict(self, batch_size: int, sequences: List[str], **kwargs):
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


def main():
    # @title Train The Model
    reset_environment()

    model = EasyT5(
        checkpoint_name=tensorboard_name,
        output_dir=LOGS_DIR, 
        learning_rate=1e-4
    )

    logger = TensorBoardLogger(
        TENSORBOARD_LOGS, 
        name=tensorboard_name
    )

    model.from_pretrained(
        model_name=model_checkpoint
    )

        # train_df= train_df.sample(100, random_state=SEED),
        # test_df = test_df.sample(100, random_state=SEED),
        # eval_df = eval_df.sample(100, random_state=SEED),
    model.train(
        train_df= preprocessed_train_df,
        test_df = preprocessed_test_df,
        eval_df = preprocessed_eval_df,
        source_max_token_len=256, 
        target_max_token_len=64, 
        batch_size=batch_size, 
        max_epochs=max_epochs, 
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        use_gpu=True,
        logger=logger,
        dataloader_num_workers=multiprocessing.cpu_count(),
    )

