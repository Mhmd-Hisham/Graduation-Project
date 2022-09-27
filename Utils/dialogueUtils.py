import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import Utils.helperFunctions as helperFunctions

get_max_category = lambda d: list(sorted(d.items(), reverse=True))[0][0]

def get_n_dialogues(df):
    return df['dialogue'].shape[0]

def get_turns_per_dialogue(df):
    """
        Returns the turns per dialogue in the given dataframe.
        The dataframe must have a 'dialogue' column.

        Returns a vector.
    """
    return df['dialogue'].astype(str).apply(lambda x: x.split("\n"))

def get_avg_turns_per_dialogue(df):
    """
        Returns the average number of turns per dialogue in the given dataframe.
        The dataframe must have a 'dialogue' column.

        Returns a scalar value.
    """
    n_dialogues = get_n_dialogues(df)
    n_turns_per_dialog = get_turns_per_dialogue(df).apply(len)
    return np.sum(n_turns_per_dialog)/n_dialogues

def get_n_turns_per_dialogue(df):
    """
        Returns the number of turns per dialogue in the given dataframe.
        The dataframe must have a 'dialogue' column.

        Returns a vector.
    """
    turns_per_dialog = get_turns_per_dialogue(df)
    n_turns_per_dialog = turns_per_dialog.apply(len)
    return n_turns_per_dialog

def get_n_words_per_dialogue(df):
    """
        Returns the number of words per dialogue in the given dataframe.
        The dataframe must have a 'dialogue' column.

        Returns a vector.
    """
    words_per_dialog = df['dialogue'].apply(lambda x: x.split())
    n_words_per_dialog = words_per_dialog.apply(len)

    return n_words_per_dialog

def get_n_turns(df):
    """
        Returns the number of turns in the 'dialogue' column of the given dataframe.
        Returns a scalar value.
    """
    return np.sum(get_n_turns_per_dialogue(df))

def get_n_words(df):
    """
        Returns the number of words in the 'dialogue' column of the given dataframe.
        Returns a scalar value.
    """
    return np.sum(get_n_words_per_dialogue(df))

def get_avg_words_per_turn(df):
    """
        Returns the average number of words per turn in the given dataframe.
        The dataframe must have a 'dialogue' column.

        Returns a scalar value.
    """
    n_words = get_n_words(df)
    n_turns = get_n_turns(df)
    return n_words/n_turns


def get_stats(df):
    stats = dict()

    stats['# dialogues'] = get_n_dialogues(df)
    stats['# utterances'] = get_n_turns(df)
    stats['avg # turns in a dialogue'] = get_avg_turns_per_dialogue(df)
    stats['avg # words in a turn'] = get_avg_words_per_turn(df)

    return stats

def get_classification_from_history(path_to_history, df, dtype='U60'):
    topics = np.array(["unclassified"]*df.shape[0], dtype=dtype)

    history = dict()
    if os.path.exists(path_to_history):
        history = helperFunctions.read_json(path_to_history)
        classified_from_history = np.array(list(history.keys())).astype(int)
        classified_from_history_values = np.array(list(map(get_max_category, 
                                                           history.values())))
        topics[classified_from_history] = classified_from_history_values

    return topics, history

def preview_dialogues(df, n=4):
    for i in range(n):
        print(df['dialogue'].iloc[i])
        print("------------------------------")

def parse_dialogue(dialogue:str, sentence_key: str='text') -> str:
    speakers = set()
    for entry in dialogue:
        speakers.add(entry['speaker'])
  
    if len(speakers) != 2:
        return ""
    
    speaker1, speaker2 = list(speakers)
    if (speaker1 != dialogue[0]['speaker']):
        speaker1, speaker2 = speaker2, speaker1

    mapper = {speaker1:"#Person1#:", speaker2:"#Person2#:"}

    output = []
    for entry in dialogue:
        speaker = entry['speaker']
        turn = mapper[speaker] + entry[sentence_key]
        output.append(turn)

    return '\n'.join(output)

class DialogueDataset:
    """
        A base class for all the dialogue datasets in the project.
    """
    def __init__(self, dataset_name: str, dataset_src_url: str, order: int):
        self.dataset_name = dataset_name
        self.dataset_src_url = dataset_src_url
        self.order = order
    
    def download(self, dest: str) -> List[str]:
        raise NotImplementedError
    
    def preprocess(self, files: List[str]) -> pd.DataFrame:
        raise NotImplementedError
    
    def get_dataset(self, dest:str, save_index: bool=False) -> Tuple[pd.DataFrame, dict]:
        files = self.download(dest)
        df = self.preprocess(files)
        info_dict = self.save(df, dest, save_index)

        return df, info_dict
    
    def save(self, df: pd.DataFrame, dest: str, save_index: bool=False) -> dict:
        """
            Saves the given pre-processed dataframe of the dataset along with the information dictionary
        """
        dest = os.path.join(dest, 'preprocessed')
        os.makedirs(dest, exist_ok=True)

        df.to_csv(os.path.join(dest, f"df{self.order}.csv"), index=save_index)

        info_dict = {
            "name": self.dataset_name,
            "source": self.dataset_src_url,
            "order": self.order
        }

        helperFunctions.save_as_json(info_dict, f"info{self.order}.json", dest)

        return info_dict
    