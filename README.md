# CNN_LSTM_Seq2Seq

Abstractive Text Summarization Using Sequence to Sequence Model

## Project Overview
Abstractive text summarization, on the other hand, generates summaries by compressing the information in the input text in a lossy manner such that the main ideas are preserved. The advantage of abstractive text summarization is that it can use words that are not in the text and reword the information to make the summarizes more readable. In this model, a CNN-LSTM encoder and LSTM decoder model are used to generate headlines for articles using the Gigaword dataset. To improve the quality of the generated summaries, a Bahdanau attention mechanism, a pointer-generator network and a beam-search inference decoder are applied to the model. 

## Install
This project requires **Python 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [PyTorch 1.0.0](https://pytorch.org/)
- [files2rouge](https://github.com/pltrdy/files2rouge)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 3.6 installer. 

## Architecture

![alt text](https://github.com/murak038/CNN_LSTM_Seq2Seq/blob/master/images/Architecture.png "Modified version of a figure shown in See et al. 2017 used to describe the overall architecture and the function of the pointer-generator. The hidden states of the LSTM encoder are used to calculate the encoder score at each timestep by using the Bahdanau attention function on the encoder hidden states and the hidden state of the previous decoder time step. The attention scores function, weighted by the probability of the pointer function, as the probability of the word in the input text being the word output by the decoder. ")

## Hyperparameters
| Parameters | Values|
| ------------- |:-------------:|
|    Kernel Size     |  [1,3,5] |
|    Filter Size     |  100 |
|    Encoder Hidden Units     |  256 |
|    Encoder Layers     |  1 |
|    Decoder Hidden Units     |  512 |
|    Decoder Layers     |  1 |
|    Beam Width     |  10 |
|    Embedding     |  300d - GloVe |
|    Dropout     |  0.5 |
|    Loss Function     |  torch.nn.CrossEntropyLoss |
|    Optimizer     |  Adam Optimizer |
|    Learning Rate     |  0.001 |

## Dataset
The model is trained on the Gigaword corpus found at https://github.com/harvardnlp/sent-summary. The dataset contains the first sentence of articles as the input text and the headlines as the ground-truth summaries. 

## Results
The generated summaries achieved a ROUGE-1 score of 29.79 using the files2rouge function. 
