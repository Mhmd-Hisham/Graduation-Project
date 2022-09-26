import os

import numpy as np
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

def parse_dialogue(dialogue:str) -> str:
    speakers = set()
    for entry in dialogue:
        speaker = entry['speaker']
        speakers.add(speaker)
  
    if len(speakers) != 2:
        return ""
    
    speaker1, speaker2 = list(speakers)
    if (speaker1 != dialogue[0]['speaker']):
        speaker1, speaker2 = speaker2, speaker1

    mapper = {speaker1:"#Person1#:", speaker2:"#Person2#:"}

    output = []
    for entry in dialogue:
        speaker = entry['speaker']
        turn = mapper[speaker] + entry['text']
        output.append(turn)

    return '\n'.join(output)