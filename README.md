# Shakespeare LSTM Text Generator

A character-level → word-level LSTM language model trained on Shakespeare's works to generate Shakespeare-like text.

<img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow"> <img src="https://img.shields.io/badge/LSTM-Recurrent%20Neural%20Network-purple?style=flat">

## What it does

This project trains a **word-level LSTM** language model on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset (~1 MB of text containing most of Shakespeare's plays and sonnets).

Given any seed phrase (one or more words, e.g. "romeo", "to be or", "my lord", "shall i compare thee"), the model predicts the most likely next word repeatedly, generating Shakespearean-style continuation text.
