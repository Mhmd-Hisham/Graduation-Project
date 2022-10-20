
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

def reset_environment(reset_seed: bool, seed: int):
    """
        Clears the memory and resets the environment seed.

        Parameters
        ----------
        reset_seed : whether to reset the environment seed or not
        seed : the seed value. Should be an integer in the range [min(np.uint32), max(np.uint32))
    """
    # empty the cuda cache
    torch.cuda.empty_cache()

    # force the python garbage collector to clear unused memory now
    gc.collect()

    # reset the seed
    if reset_seed:
        pl.seed_everything(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

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
        super().__init__(self)
        for k, v in params.items():
            self[k] = v
        return self

    def to_json(self, filepath: str):
        """
            Saves the experiment parameters to a JSON file.
        """
        helperFunctions.save_as_json(self, filepath, "")

    def __str__(self):
        return json.dumps(self, indent=2)

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

        # since those parameters are used in every __getitem__ call,
        # It's better to store the parameters in the class for faster extraction
        # as dictionaries are slightly slower 
        self.tokenizer_max_length = params['encoding']['max_length']
        self.tokenizer_padding = params['encoding']['padding']
        self.tokenizer_truncation = params['encoding']['truncation']
        self.tokenizer_add_special_tokens = params['encoding']['add_special_tokens']

    def __len__(self) -> int:
        """ returns the length of data """
        return self.data.shape[0]

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5 model"""

        row = self.data.iloc[index]

        encoded_source = self.tokenizer(
            row.source_text,
            max_length=self.tokenizer_max_length,
            padding=self.tokenizer_padding,
            truncation=self.tokenizer_truncation,
            add_special_tokens=self.tokenizer_add_special_tokens,

            return_attention_mask=True,
            return_tensors="pt",
        )

        encoded_target = self.tokenizer(
            row.target_text,
            max_length=self.tokenizer_max_length,
            padding=self.tokenizer_padding,
            truncation=self.tokenizer_truncation,
            add_special_tokens=self.tokenizer_add_special_tokens,

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

        self.num_workers = params['general']['cpu_cores']
        self.batch_size = params['trainer']['batch_size']

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

class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model"""

    def __init__(
        self,
        tokenizer,
        model,
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
        self.tensorboard_name = params['general']['tensorboard_name']
        self.tokenizer = tokenizer
        self.params = params

        self.save_last_n_epochs = params['trainer']['save_last_n_epochs']
        self.saved_epochs = []

        self.average_training_loss = None
        self.average_validation_loss = None

        self.checkpoints_dir = f"{self.params['general']['output_dir']}/{self.tensorboard_name}/"

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
            lr=self.params['trainer']['fixed_learning_rate']
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

class EasyT5:
    """ Custom class for fine-tunning/training T5 models"""

    def __init__(self, params: ExperimentParameters):
        """ Initializes the EasyT5 class """
        self.trainer = None
        self.callbacks = [TQDMProgressBar(refresh_rate=5)]

        self.params = params
        self.checkpoint_name = params['general']['checkpoint_name']
 
    def from_pretrained(
        self,
        tokenizer_class: PreTrainedTokenizer,
        model_class: PreTrainedModel,
        return_dict: bool=True,
        use_gpu: bool=True,
        fp16: bool=False) -> None:
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
                monitor=self.params['trainer']['early_stopping_monitor'],
                min_delta=self.params['trainer']['early_stopping_min_delta'],
                patience=self.params['trainer']['early_stopping_patience_epochs'],
                mode=self.params['trainer']['early_stopping_mode'],
                verbose=True,
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

    def predict(self, source_text: str, custom_params: ExperimentParameters=None) -> List[str]:
        """
            Generates predictions
            Parameters
            ----------
                source_text (str): any text for generating predictions
                custom_params : custom parameters for the output generation. The default is to use the parameters provided when the class was initialized.

            Returns
            -------
                list[str]: returns the predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )

        params = self.params if custom_params == None else custom_params

        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=params['generator']['num_beams'],
            max_length=params['generator']['max_length'],
            repetition_penalty=params['generator']['repetition_penalty'],
            length_penalty=params['generator']['length_penalty'],
            early_stopping=params['generator']['early_stopping'],
            top_p=params['generator']['top_p'],
            top_k=params['generator']['top_k'],
            num_return_sequences=params['generator']['num_return_sequences'],
        )

        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=params['generator']['skip_special_tokens'],
                clean_up_tokenization_spaces=params['generator']['clean_up_tokenization_spaces'],
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

def main():
    SEED = 1234
    parameters = ExperimentParameters()

    # parameters related to the training process
    # and the PyTorch Lightning trainer
    parameters['trainer'] = {
        # saves the last recent 'n' epochs
        "save_last_n_epochs": 3,
        # the fixed learning rate for the model
        "fixed_learning_rate": 1e-4,
        # the monitor of the early stopping
        "early_stopping_monitor": "val_loss",
        # the minimum delta between the epochs to apply early stopping
        "early_stopping_min_delta": 0.01,
        # 0 to disable early stopping feature
        "early_stopping_patience_epochs": 0,
        # the mode of the early stopping criteria
        "early_stopping_mode": "min",
        # the maximum number of epochs to train/fine-tune the model on
        "max_epochs": 5,
        # the floating point numbers precision
        "precision": 32,
        # the training batch size 
        # the batch size at which the data is loaded into memory
        "batch_size": 8,
    }

    # general parameters about the working environment
    parameters['general'] = {
        # the output directory
        "output_dir":"",
        # the name/path of the checkpoint to be loaded from Hugging face
        # or from the local disk
        "checkpoint_name":"",
        # the name that will appear on tensorboard
        "tensorboard_name": "",
        # the number of cpu cores in the current machine
        "cpu_cores": multiprocessing.cpu_count(),
        # the environment seed
        'seed':SEED,
    }

    # the parameters passed to the tokenizer when encoding text
    parameters['encoder'] = {
        # the padding method for the input sequences
        "padding":"max_length",
        # whether to truncate long sequences or not
        "truncation":True,
        # whether to add special tokens in the input sequences or not
        "add_special_tokens": True,
        # the maximum length of the input sequence
        "max_length": 512,
    }

    # the parameters passed to the model when generating text
    parameters['generator'] = {
        # the number of beams used in the beam search (also known as beam width)
        "num_beams": 2,
        # the maximum length of the generated sequences
        "max_length": 512,
        # the repetition penalty added when the model repeats words
        "repetition_penalty": 2.5,
        # the penalty aded when the model generates lengthly sequences
        "length_penalty": 1.0,
        # whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        "early_stopping": True,
        # if set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
        "top_p": 0.95,
        # the number of highest probability vocabulary tokens to keep for top-k-filtering.
        "top_k": 50,
        # the number of returned sequences
        "num_return_sequences": 1,
        # whether to skip special tokens when generating or not
        "skip_special_tokens": True,
        # whether to clean all tokenization spaces before returning the output or not
        "clean_up_tokenization_spaces": True,
    }

    ################ model training ################
    # NOT TESTED YET :(
    reset_environment(seed=parameters['general']['seed'])

    # loading the model
    model = EasyT5(parameters)
    model.from_pretrained(T5ForConditionalGeneration, T5Tokenizer)

    # setting up the tensorboard logger
    TENSORBOARD_LOGS = "path_to_tensorboard_logs_dir"
    logger = TensorBoardLogger(
        TENSORBOARD_LOGS, 
        name=parameters['general']['tensorboard_name']
    )

    # load the dataframes
    preprocessed_train_df = None
    preprocessed_test_df = None
    preprocessed_eval_df = None

    # train the model
    model.train(
        train_df=preprocessed_train_df,
        test_df=preprocessed_test_df,
        eval_df=preprocessed_eval_df,
        accelerator='gpu',
        logger=logger,
    )
    ################################################


    ############### model inference ################
    model = EasyT5(parameters)
    model.from_pretrained(T5Tokenizer, T5ForConditionalGeneration, return_dict=False)
    suggestion = model.predict("complete: I want to eat")[0]
    print(suggestion)
    ################################################
