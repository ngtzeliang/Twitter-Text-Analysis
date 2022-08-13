## The Analytics Edge Summer 2022 Data Competition

Competition code for "The Analytics Edge" module in SUTD. Original dataset can be found from [Kaggle](https://www.kaggle.com/c/2022tae/leaderboard).

### For code submission on eDimension

You will need to download the whole folder and reference that folder's name when importing the model. There is no need to download this unless the exact predictions need to be reproduced.

### Overview

Our intial dataset consisted of 20610 Tweets gathered from Twitter. The task was to develop an algorithm that determines, with the highest accuracy, the general sentiment of the tweets. Specifically, the challenge is to determine whether each tweet has a negative, neutral, or positive sentiment.


### Approach

Our approach was to use transfer learning from pre-trained models. Various advances in the field of Natural Language Processing have yielded highly effective pre-trained models that were built on the idea of handling text as sequential data. Hence, we implemented the usage of Recurrent Neural Networks (RNN), more specifically, the Long Short-Term Memory (LSTM) RNN.

For our case, the pre-trained models selected were Facebook’s [fastText](https://fasttext.cc/) and Stanford University's [GloVe](https://nlp.stanford.edu/projects/glove/). In a nutshell, the corpus was separated into training and validation sets. The models allowed us to create an algorithm for obtaining vector representations for words. 

The model itself was fine-tuned to the task at hand by, firstly, passing the input data into the pre-trained model. The mean of its output was then passed through two fully-connected layers (32-dimensional each) with ReLU activation. The output layer of the model was three-dimensional with the Softmax activation. The dimension with the highest probability denotes the sentiment class. To compute loss, the categorical cross-entropy loss was used along with the Adam optimiser for the gradient-based optimisation. The model was run for 20 epochs, since the validation loss increased very slightly at around the 20th epoch, in general.

While training the model, we noticed that our predicted accuracy is bounded by the limited 20610 tweets. Therefore, we went to gather more data from [Kaggle](https://www.kaggle.com/competitions/tweet-sentiment-extraction) and augmented the data to our initial dataset. This brought up the size of our dataset to 51624 tweets.

Due to the stochastic nature of the neural network model, the exact model, and hence, the resulting predictions will differ slightly with different runs. As such, the final model used is freezed and can be reloaded to produce the same predictions as the one submitted on Kaggle. 

### Dependencies

An R (2022.7.1.554) environment is required, preferably with Tensorflow, Keras, and Reticulate installed already. If you do not have them installed, 

1. Install Conda. Preferably, a light install using Miniconda is preferred.
2. Open RStudio.

We also downloaded a 100-dimensional fastText pre-trained word embeddings and a 200-dimensional GloVe pre-trained word embeddings for our model. 

### R Packages Used

- `tidyverse`
- `caTools`
- `keras`
- `reticulate`
- `tm`
- `textclean`
- `caret`
- `dplyr`
- `tensorflow`

### Results

The model was able to predict the sentiments of the tweets with a training accuracy of around 87.79%, a validation accuracy of around 82.98%, and 87.384% in the Kaggle data competition’s public leaderboard. As mentioned above, due to the stochastic nature of the neural network, we can expect to see some slight differences in accuracy with each run of the script. 

### Interpretability and Limitations

Although the model was able to predict the sentiments to a respectable degree of accuracy, our understanding of exactly how LSTM RNN works, at this point, is limited. The neural network made the modelling process at times akin to a black box algorithm, where debugging and manual tuning were more challenging. 

As for the limitations of the model, one possible limitation of using the pre-trained models is that the model is pre-trained on unfiltered content from the internet, where neutrality is not guaranteed. This could result in the possibility of the model having more biased predictions compared to a model pre-trained on data from another source. Furthermore, even after augmenting more tweets to our original dataset, the dataset of 51624 tweets was still insufficient and hence limited our prediction accuracy. 
