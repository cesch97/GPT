GPT
===
The project is a simple implementation of a Transformer model for text generation. The model is implemented from scratch in PyTorch and trained on a conversational dataset with AWS Sagemaker. 

### Model
The model is a simple Transformer as described in the original paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). It contains all the main components of the Transformer model: multi-head self-attention, feed-forward neural network, and positional encoding. The model is trained on a conversational dataset and is able to generate text based on the input prompt.

### Data
The model is trained on a conversational dataset composed of a collections of conversations between two agents. The dataset is preprocessed by extracting all continuous sequences of characters into words and selecting the most frequent words for the vocabulary. Then the dataset is tokenized by replacing each word with its corresponding index in the vocabulary and including special tokens for padding, start of sentence, end of sentence and unknown words. The model is then trained to take both side of the conversations.

### Training
The model can be trained locally or on AWS Sagemaker. The training script is implemented in PyTorch and allows to keep checkpoints of the model during training and restart training from a checkpoint. The training script is also able to save the model and the vocabulary to be used for inference.

### Inference
The model can be used for inference by loading the model and the vocabulary from the training script. The model is able to generate text based on a given prompt and keep track of the context of the conversation. The output of the model is a probabilty distribution over the vocabulary that can be used to sample the next word based on the temperature parameter.

### Results
The model was trained for 24 hours on a *ml.p2.xlarge* instance on AWS Sagemaker. I noticed a progrssive improvement in the quality of the generated text going from complete randomness to meaningful outputs. Still the model was not able to generate coherent conversations and often the response was not related to the input prompt.

### Usage
Running the training script on AWS Sagemaker does not require any additional libraries. To run the project on Sagemaker upload the project folder to a SageMaker instance, uplade the dataset to an S3 bucket and prepare a configuration file (see `config/sagemaker_config.json` for an example). At the end of training or if the number of ecpochs in the configuration file is lower than the number of epochs in the checkpoint the model will be automatically run in inference mode allowing to have a conversation with the model.