
import os
import sys

import streamlit as st
from streamlit_chat import message
PROJECT_PATH = "/content/drive/MyDrive/TWM/Graduation-Project/"
PROJECT_PATH = rf"C:\Users\{os.environ['USERNAME']}\Graduation-Project"

# number of suggestions
num_return_sequences = 3

def load_best_t5_model():
    from Utils.EasyT5 import EasyT5, ExperimentParameters
    from transformers import T5ForConditionalGeneration
    from transformers import T5TokenizerFast as T5Tokenizer
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

    # load the model
    model = EasyT5(parameters)
    model.from_pretrained(T5Tokenizer, T5ForConditionalGeneration, return_dict=False)

    return model

model = load_best_t5_model()

# st.text('Fixed width text')
# st.markdown('_Markdown_') # see *
st.title('GP Streamlit Interface')

# text_area = st.text_area('Conversation history', disabled=True)

prompt = st.text_input('Enter prompt:')
if prompt:
    message(prompt, is_user=True)
    suggestion = model.predict("complete: " + prompt)[0]
    message(suggestion, is_user=False)
