

import time
import os
from typing import List

import streamlit as st
from streamlit_searchbox import st_searchbox

PROJECT_PATH = "/content/drive/MyDrive/TWM/Graduation-Project/"
PROJECT_PATH = rf"C:\Users\{os.environ['USERNAME']}\Graduation-Project"

# number of suggestions
num_return_sequences = 3
MODEL = None

@st.experimental_singleton
def load_best_t5_model():
    from Utils.EasyT5 import EasyT5, ExperimentParameters, reset_environment
    from transformers import T5ForConditionalGeneration
    from transformers import T5TokenizerFast as T5Tokenizer
    reset_environment(reset_seed=True, seed=512)

    MODEL_CHECKPOINTS = "FineTuning/T5/checkpoints/"

    parameters = ExperimentParameters().from_json(
        os.path.join(
            MODEL_CHECKPOINTS,
            "parameters.json"
        )
    )

    parameters['general']["checkpoint_name"] = os.path.join(
    PROJECT_PATH,
    MODEL_CHECKPOINTS,
    "-epoch-9-tloss-1.5577-vloss-1.8296"
    )

    parameters['general']["tensorboard_name"] = "t5-v1_1-base_BatchSize-16_N-Splits-4_DatasetSize-large_Topic-Food&Drink_version_2"
    parameters['generator']['num_return_sequences'] = num_return_sequences
    parameters['generator']['num_beams'] = num_return_sequences

    # load the model
    model = EasyT5(parameters)
    model.from_pretrained(T5Tokenizer, T5ForConditionalGeneration, return_dict=False)

    return model

def debounce(threshold):
    """
        A simple decorator to implement debouncing in order to prevent unnecessary calls to the model.
        Since the application is real time, the model needs to be called per every new word written, 
        but the input calls the prediction function per key stroke. 
        
        The debounce decorator aims to limit the number of calls until the user finishes the whole input prompt.
        The threshold value is the number of seconds to wait between subsequent function calls.
    """

    # dirty trick to access the variable by reference
    t0 = [time.time()]
    def inner(func):
        def __core__(*args, **kwargs):
            elapsed = time.time() - t0[0]
            if elapsed < threshold:
                suggestions = func(*args, **kwargs)
                t0[0] = time.time()
                return suggestions

            else:
                return []
    
        return __core__

    return inner

@debounce(threshold=0.3)
def get_suggestions(prompt) -> List[str]:
    if prompt:
        suggestions = MODEL.predict_multiple("complete: " + prompt)
        suggestions = [prompt + s for s in suggestions]
        return suggestions
    return []

# it's loaded only once, thanks to the singleton decorator
MODEL = load_best_t5_model()

st.title('GP Streamlit Interface')

history_text = st.text_area(
    label="Conversation history",
    placeholder="conversation history appears here...",
    disabled=True,
)

chosen_suggestion = st_searchbox(
    get_suggestions,
    key="chatbot_suggestions",
)


# st.markdown("You've selected: %s" % selected_value)