
## Description

In this project, I collect, clean, pre-process, augment, and build a high-quality dialogue dataset with a large distribution of topics.
The goal was to train/fine-tune large-scale language models to predict dialogue-based conversations and to use the top language model to make real-time reply suggestions in a chat app. The feature is similar to Google's [SmartCompose](https://ai.googleblog.com/2018/05/smart-compose-using-neural-networks-to.html).

As of now, the dataset was only used to fine-tune [Google's T5 model](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html). 

~~This was the final [tensorboard](https://tensorboard.dev/experiment/AK2Q5596RxC32lzooQVGSA/#scalars) for the project.~~ TensorBoard.dev was shut down as of January 1, 2024, and the tensorboard log data were lost. Luckily, I had a copy of the original notebook, which ran on Colab Pro+ and had tensorboard visualizations embedded in it.

You can view the original notebook with the live tensorboard on Colab from [here](https://colab.research.google.com/drive/12s7ArkjW9IumtCK17raKZhRzaBHtGczK?usp=sharing).

Here's a copy of the final graduation project [presentation](https://docs.google.com/presentation/d/1q0D84YZLrT0_jbr7zmd1Ta6JfqEQ23fmjqXfbdTpjFI/edit?usp=sharing). I was responsible for delivering the whole data science part of the project.


## Requirements

Create a virtual environment:

```
python -m venv chatbot-env
```

Activate it:
```
./chatbot-env/Scripts/activate.bat # In CMD
./chatbot-env/Scripts/Activate.ps1 # In Powershel
./chatbot-env/Scripts/activate     # In linux/Mac OS X
```

Install the requirements:
```
python -m pip install -r requirements.txt
```

Notes:

 - Currently, I run the project locally on python 3.9. However, I might consider downgrading to 3.7 to match the current version of python on Google Colab.
 
 - I use PyTorch with Cuda v11.3 (same as the one currently on Google Colab). If you have a different Cuda version installed, you might need to update the requirements file. Have a look at [this](https://pytorch.org/get-started/locally/).


## Meta

Mohamed Hisham – [Gmail](mailto:Mohamed00Hisham@Gmail.com) | [GitHub](https://github.com/Mhmd-Hisham) | [LinkedIn](https://www.linkedin.com/in/Mhmd-Hisham/)


