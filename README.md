# time-series-transformer

Trying out the transformer model for time-series. Mainly proof of concept. Training locally with pytorch - extend to train in Azure cloud.

Implemented:
-Encoder-only w/ generative training
-Encoder-only w/ Bert-style training
-Transformer (encoder + decoder)
-Electra https://arxiv.org/pdf/2003.10555.pdf

ToDo: 
Implement inference for the electra model.
Explore the informer model.
