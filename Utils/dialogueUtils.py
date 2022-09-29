
import re
import os
import random
from typing import List, Optional, Tuple, Dict

from google.cloud import language_v1
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

def remove_speaker_prefix(dialogue: str) -> str:
    # removes "#Person1#: ", "#Person2#: ",
    #         "#Person1#:",  "#Person2#:"
    return re.sub('(\#Person[1|2]\#\:\s*)', "", dialogue)

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


def classify_dialogue(dialogue: str, verbose: bool=True) -> Dict[str, float]:
    """
        Classify the category of the input text using Google NLU APIs.
        API Docs: https://cloud.google.com/natural-language/docs/quickstarts
        
        https://cloud.google.com/natural-language/docs/sentiment-analysis-client-libraries
    """

    result = {}
    # return result
    
    language_client = language_v1.LanguageServiceClient()

    document = language_v1.Document(
        content=dialogue, type_=language_v1.Document.Type.PLAIN_TEXT
    )
    response = language_client.classify_text(request={"document": document})
    categories = response.categories

    for category in categories:
        # Turn the categories into a dictionary of the form:
        # {category.name: category.confidence}, so that they can
        # be treated as a sparse vector.
        result[category.name] = category.confidence

    return result

# the API returns multiple categories, each with a confidence score
# given the API dictionary, this function returns the max category 
get_max_category = lambda d: list(sorted(d.items(), reverse=True))[0][0]

# uncomment to test the classify_dialogue function
# df = pd.read_csv("DataEngineering/Datasets/dataset1/preprocessed/df1.csv")
# classify_dialogue(df.iloc[4]['dialogue'])

def classify_dataset(dataset_index: int, percentage: float, save_threshold: int, seed: int):
    """
        Loads the given dataset and classifies all the dialogues using Google APIs.
        It tries to use cashed API calls (if any).

        Parameters:
        :param dataset_index: the index of the dataset. can be found in the 'order' value of the dataset info dictionary.
        :param percentage: the percentage in which we classify the dataset. A percentage of 1 means classify all the dialogues. At some point, only a random sample of each dataset was being classified for testing purposes
        :param save_threshold: save API call results to disk every 'save_threshold' calls
    """
    # each time the datasets get shuffled and a random sample is classified
    # reset the seed since the datasets were partially classified at some point
    random.seed(seed)

    dataset_path = f"DataEngineering/Datasets/dataset{dataset_index}/preprocessed/"
    csv_path = dataset_path + f"df{dataset_index}.csv"

    history_file = f"classification_history{dataset_index}.json"
    history_path = dataset_path + history_file

    # load classification history (if exists) for this dataset 
    history = dict() if not os.path.exists(history_path) else \
                helperFunctions.read_json(history_path)

    # convert keys to integers
    history = {int(k):v for k,v in history.items()}

    # load the dataset dataframe
    df = pd.read_csv(csv_path)
    df = df[df['dialogue'].notna()]

    # populate the classified entries from the history object
    topics = np.array(["unclassified"]*df.shape[0], dtype='U60')
    if len(history):
        classified_from_history = np.array(list(history.keys()))
        classified_from_history_values = np.array(list(map(get_max_category, 
                                                            history.values())))
        topics[classified_from_history] = classified_from_history_values

    # compute the number of dialogues to classify based on the percentage
    sample_size = int(percentage * df.shape[0])

    # fetch the sample rows
    sample_rows = random.sample(range(df.shape[0]), sample_size)

    # print("Classification Percentage: ", len(history)/len(topics))
    for j, row in enumerate(sample_rows, 1):
        if len(history)/len(topics) >= percentage:
            break
        
        print(f"\rProcessing row {j} [{round(100*j/sample_size,2)}%]", end="")

        # already classified and stored in the cache, skip
        if row in history and history[row]!="error":
            # print(history[row])
            continue
        
        dialogue = df.iloc[row]['dialogue']
        
        classification = {"error": 1.0}
        try:
            # entries with less than 80 characters are mostly rejected by the API
            # so I decided to discard such entries to save API calls
            if len(dialogue)>80:
                classification = classify_dialogue(dialogue, False)
                if not classification:
                    classification = {"unknown": 1.0}

        except Exception as error:
            print(f"\rError on row {j}:", error, "Dialogue length:", len(dialogue))
        
        history[row] = classification
        topics[row] = get_max_category(classification)

        if j%save_threshold == 0:
            # helperFunctions.save_as_json(history, history_file, dataset_path)
            pass

        # limit API calls to 16 call per second (960 call per minute)
        # this was important since I didn't want to transition into a higher pricing segment
        # time.sleep(1/16)

    df["Google Classification"] = topics

    # cache the API calls after finishing
    helperFunctions.save_as_json(history, history_file, dataset_path)

    return df, history