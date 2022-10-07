
#@markdown Importing modules

import os
import gc
import glob
import json
import shutil
import random
import multiprocessing
from typing import List, Tuple, Dict, Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import T5TokenizerFast as T5Tokenizer

import Utils.helperFunctions as helperFunctions

SEED = 1234
def reset_environment(reset_seed=True, seed=SEED):
    """
        Clears the memory and resets the environment seed
    """
    torch.cuda.empty_cache()
    gc.collect()
    if reset_seed:
        pl.seed_everything(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

class ExperimentParameters(dict):
    """
        A class for serializing/deserializing deep learning experiment parameters.
        It's basically a wrapper class for JSON objects.
    """
    def __init__(self):
        super().__init__(self)

    def from_json(self, filepath: str):
        """
            Loads the experiment parameters from a JSON file.
        """
        params = helperFunctions.read_json(filepath)
        super().__init__(self, params)

    def to_json(self, filepath: str):
        """
            Saves the experiment parameters to a JSON file.
        """
        helperFunctions.save_as_json(self, filepath, "")

    def __str__(self):
        return json.dumps(self, indent=2)

#@title LightningDataset
class LightningDataset(Dataset):
    """  PyTorch Dataset class  """
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        params: ExperimentParameters,
    ):
        """
        Initiates a PyTorch Dataset Module for input data

        Parameters
        ----------
            data : input pandas dataframe. Dataframe must have 2 column: "source_text" and "target_text"
            tokenizer : a PreTrainedTokenizer from hugging face.
            params : a nested dictionary that contains all the parameters for encoding/decoding 
        """
        self.data = data
        self.tokenizer = tokenizer
        self.encoding_params = params['encoding']

    def __len__(self) -> int:
        """ returns the length of data """
        return self.data.shape[0]

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5 model"""

        row = self.data.iloc[index]

        encoded_source = self.tokenizer(
            row.source_text,
            max_length=self.encoding_params['max_length'],
            padding=self.encoding_params['padding'],
            truncation=self.encoding_params['truncation'],
            add_special_tokens=self.encoding_params['add_special_tokens'],

            return_attention_mask=True,
            return_tensors="pt",
        )

        encoded_target = self.tokenizer(
            row.target_text,
            max_length=self.encoding_params['max_length'],
            padding=self.encoding_params['padding'],
            truncation=self.encoding_params['truncation'],
            add_special_tokens=self.encoding_params['add_special_tokens'],

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
        params: ExperimentParameters,
    ):
        """
        initiates a PyTorch Lightning Data Module

        Parameters
        ----------
            train_df : the training dataframe. 
            test_df : the test dataframe. 
            eval_df : the validation dataframe. 
            tokenizer: a PreTrainedTokenizer object from hugging face
            batch_size: batch size of loading the data.

            The dataframes must have only two columns: "source_text" and "target_text"
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.eval_df = eval_df
        self.tokenizer = tokenizer
        self.params = params

        self.num_workers = params['other']['cpu_cores']
        self.batch_size = params['other']['data_module_batch_size']

    def setup(self, stage=None):
        self.train_dataset = LightningDataset(self.train_df, self.tokenizer, self.params)
        self.test_dataset = LightningDataset(self.test_df, self.tokenizer, self.params)
        self.val_dataset = LightningDataset(self.eval_df, self.tokenizer, self.params)

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
    """ PyTorch Lightning Model"""

    def __init__(
        self,
        tokenizer,
        model,
        checkpoint_name: str,
        params: ExperimentParameters,
    ):
        """
            Initializes the PyTorch Lightning Model

            Parameters
            ----------
            tokenizer : the T5 tokenizer, loaded from hugging face.
            model : the T5 model, loaded from hugging face.
            checkpoints_dir : output directory to save model checkpoints.
            checkpoint_name : the name of the model checkpoint
            params : the experiment parameters 
        """
        super().__init__()
        self.model = model
        self.checkpoint_name = checkpoint_name
        self.tokenizer = tokenizer
        self.params = params

        self.save_last_n_epochs = params['other']['save_last_n_epochs']
        self.saved_epochs = []

        self.average_training_loss = None
        self.average_validation_loss = None

        self.checkpoints_dir = f"{self.params['general']['output_dir']}/{self.checkpoint_name}/"

        # gets the next experiment version
        self.experiment_version = list(
            sorted(
                map(lambda s: int(s[s.rfind('version')+8:]),
                   ['version_-1']+glob.glob(self.checkpoints_dir+"/version_*")),
                   reverse=True)
            )[0]+1

        self.checkpoints_dir = os.path.join(self.checkpoints_dir, f"version_{self.experiment_version}/")

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ Forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """ Training step """
        loss, outputs = self(
            input_ids=batch["source_text_input_ids"],
            attention_mask=batch["source_text_attention_mask"],
            decoder_attention_mask=batch["labels_attention_mask"],
            labels=batch["labels"],
        )

        self.log(
            "Loss/train", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def validation_step(self, batch, batch_size):
        """ Validation step """
        loss, outputs = self(
            input_ids=batch["source_text_input_ids"],
            attention_mask=batch["source_text_attention_mask"],
            decoder_attention_mask=batch["labels_attention_mask"],
            labels=batch["labels"],
        )

        self.log(
            "Loss/validation", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def test_step(self, batch, batch_size):
        """ Test step """
        loss, outputs = self(
            input_ids=batch["source_text_input_ids"],
            attention_mask=batch["source_text_attention_mask"],
            decoder_attention_mask=batch["labels_attention_mask"],
            labels=batch["labels"],
        )

        self.log("Loss/test", loss, prog_bar=True, logger=True,)
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
            lr=self.params['optimizer']['fixed_learning_rate']
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

    def __init__(self, params: ExperimentParameters):
        """ Initializes the EasyT5 class """
        self.trainer = None
        self.callbacks = [TQDMProgressBar(refresh_rate=5)]

        self.params = params
        self.checkpoint_name = params['model']['checkpoint_name']
 
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
            self.model = self.model.half()

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        logger="default",
        accelerator: str='gpu',
    ):
        """
            Trains the model on custom a dataset
            Parameters
            ----------
            train_df (pd.DataFrame): the training dataframe. 
            test_df ([type], optional): the test dataframe. 
            eval_df ([type], optional): the validation dataframe. 
            accelerator (str, optional): which accelerator to use when training the model. Defaults to 'gpu'.
            logger (pytorch_lightning.loggers) : any logger supported by PyTorch Lightning. Defaults to "default". If "default", pytorch lightning default logger is used.

            The dataframes must have only two columns: "source_text" and "target_text"
        """
        self.data_module = LightningDataModule(
            train_df,
            test_df,
            eval_df,
            self.tokenizer,
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

        # add logger
        loggers = True if logger == "default" else logger

        # prepare trainer
        self.trainer = pl.Trainer(
            logger=loggers,
            callbacks=self.callbacks,
            max_epochs=self.params['trainer']['max_epochs'],
            precision=self.params['trainer']['precision'],
            accelerator=accelerator,
            log_every_n_steps=1,
        )

        # fit trainer
        self.trainer.fit(self.lightning_model, self.data_module)

    def predict(self, source_text: str) -> List[str]:
        """
            Generates predictions
            Parameters
            ----------
                source_text (str): any text for generating predictions

            Returns
            -------
                list[str]: returns the predictions
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

    def predict_multiple(self, source_text: List[str]) -> List[str]:
        """
            Predict multiple samples at once.
        """
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

        return [
            self.tokenizer.decode(
                gs, 
                skip_special_tokens=self.params['generator']['skip_special_tokens'], 
                clean_up_tokenization_spaces=self.params['generator']['clean_up_tokenization_spaces']
            ) for gs in generated_samples
        ]

    def batch_predict(self, batch_size: int, sequences: List[str]) -> List[str]:
        """
            Computes the predictions on batches and then concatenates.
        """
        output = []
        n_steps = (len(sequences)//batch_size)

        # divide the input into batches and predict each batch on it's own
        for i in range(0, n_steps*batch_size, batch_size):
            output += self.predict_multiple(sequences[i:i+batch_size])

        # add any leftover samples to a separate batch
        output += self.predict_multiple(sequences[n_steps*batch_size:])

        # return all concatenated predictions
        return output


def main1():
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


def main():
    # @title Train The Model
    reset_environment()

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

