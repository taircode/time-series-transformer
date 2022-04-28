# time-series-transformer

Trying out the transformer model for time-series. Mainly proof of concept. Training locally with pytorch - extend to train in Azure cloud.

Implemented:
- Encoder-only
- Encoder-only w/ Bert-style training
- Transformer (encoder + decoder)
- Electra https://arxiv.org/pdf/2003.10555.pdf

ToDo: 
- Implement inference for the electra model.
- Explore the informer model. https://arxiv.org/pdf/2012.07436.pdf
- Implement Pegasus-style and Pegasus/Electra hybrid. https://arxiv.org/pdf/1912.08777.pdf

|  | Masked | Discriminator
------------- | ------------- | -------------
1 token | Bert  | Electra
\>1 token  | Pegasus  | Pegasus/Electra
