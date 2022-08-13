rm(list=ls())

library(tidyverse) # To load dplyr, ggplot2
library(caTools) # For sample.split
library(keras) # To load the neural network package
library(reticulate) # To enable python calling when using Keras
library(randomForest) # For Random Forest
library(e1071) # For Naive Bayes Classification
library(rpart) # For Classification Tree
library(pROC) # For multiclass ROC
library(tm) # To load text mining package
library(textclean) # To load text cleaning package
library(caret)
library(dplyr)
library(tensorflow)

train <- read.csv("new_train.csv", stringsAsFactors=FALSE)
test <- read.csv("test.csv", stringsAsFactors=FALSE)

train <- train %>% mutate(Split = "train")
test <- test %>% mutate(Split = "test")

# Combine data for tokenization
full <- data.frame(rbind(train %>% select(-sentiment), test %>% select(-id)))

# Process the Text
raw_text <- full$tweet

# Using textclean library
processed_text <- raw_text %>%
  tolower() %>%
  replace_word_elongation() %>%
  replace_internet_slang() %>%
  replace_emoticon() %>%
  replace_url() %>%
  replace_email() %>%
  replace_html()

full_processed <- full
full_processed$tweet <- processed_text

max_words <- 15000

# Prepare to tokenize the text
texts <- full_processed$tweet

tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)

sequences <- texts_to_sequences(tokenizer, texts)
word_index <- tokenizer$word_index

# Length of sequence
maxlen <- max(as.numeric(summary(sequences)[,1]))

# Pad out texts so everything is the same length
data <- pad_sequences(sequences, maxlen = maxlen)



# Split back into train and test
train_matrix <- data[1:nrow(train),]
test_matrix <- data[(nrow(train)+1):nrow(data),]

# Training labels (1, 2, 3) - One-hot encoding of train_labels
train_labels <- train$sentiment
train_labels <- train_labels %>%  data.frame() %>%
  mutate(
    V1 = ifelse(train_labels == 1, 1, 0),
    V2 = ifelse(train_labels == 2, 1, 0),
    V3 = ifelse(train_labels == 3, 1, 0),
  ) %>% 
  select(
    V1,V2,V3
  ) %>% as.matrix()

set.seed(123)
trainid <- sample.split(train$sentiment, 0.8)

x_train <- train_matrix[trainid,]
y_train <- train_labels[trainid,]

x_val <- train_matrix[!trainid,]
y_val <- train_labels[!trainid,]



# Pre-Trained Word Embedding
glove_twitter_embedding_dim <- 200
fasttext_twitter_embedding_dim <- 100
glove_twitter_weights <- readRDS("glove_twitter_27B_200d.rds")
fasttext_twitter_weights <- readRDS("fasttext_english_twitter_100d.rds")

# Input layer
input <- layer_input(
  shape = list(NULL),
  dtype = "int32",
  name = "input"
)


################### Simple LTSM with Word Embedding ######################
# Glove Embedding and LSTM layer 1
encoded_1 <- input %>%
  layer_embedding(name = "glove_twitter_embedding",
                  input_dim = max_words, 
                  output_dim = glove_twitter_embedding_dim, 
                  input_length = maxlen) %>%  
  layer_lstm(units = 32, 
             input_shape = c(maxlen, glove_twitter_embedding_dim), 
             return_sequences = FALSE)

#Fasttext Embedding and LSTM layer 2
encoded_2 <- input %>%
  layer_embedding(name = "fasttext_twitter_embedding",
                  input_dim = max_words,
                  output_dim = fasttext_twitter_embedding_dim,
                  input_length = maxlen) %>%
  layer_lstm(units = 32,
             input_shape = c(maxlen, fasttext_twitter_embedding_dim),
             return_sequences = FALSE)

model3 <-keras_model(input,(
  layer_concatenate(list(encoded_1,encoded_2)) %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = 3, activation = "softmax")))

# Set the weights to the pretrained word embedding weights
# get_layer(model3, name = "glove_twitter_embedding") %>%
#   set_weights(list(glove_twitter_weights)) %>%
#   freeze_weights()
# # 
# get_layer(model3, name = "fasttext_twitter_embedding") %>%
#   set_weights(list(fasttext_twitter_weights)) %>%
#   freeze_weights()

summary(model3)

# Compile model

model3 %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.0020),
  loss = "categorical_crossentropy",
  metrics = "categorical_accuracy"
)

# Fit model 
history3 <- model3 %>% fit(
  x_train,
  y_train,
  batch_size = 64,
  validation_data = list(x_val, y_val),
  epochs = 20,
  view_metrics = FALSE,
  verbose = 0
)

print(history3)
plot(history3)

res <- predict(model3,test_matrix)
newres <- max.col(res)
submission <- data.frame(Id = 1:length(newres), sentiment = newres)
write.csv(submission, file = "predictions.csv", row.names = F)




