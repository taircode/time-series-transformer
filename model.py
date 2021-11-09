import torch
import torch.nn as nn
import positional

class myEncoder(nn.Module):
    def __init__(self, d_model: int=512, num_layers: int=4, nhead: int=1, embed_true: bool=True, peconcat_true: bool=True, num_ts_in: int=1, num_ts_out: int=1, pe_features: int=10, seq_length: int=100):
        super().__init__()

        #self.num_ts_in=num_ts_in
        self.peconcat_true=peconcat_true
        self.embed_true=embed_true
        self.seq_length=seq_length

        if self.peconcat_true:
            self.embedding_dim=num_ts_in+pe_features
            self.pe_features=pe_features
        else:
            self.embedding_dim=num_ts_in
            self.pe_features=num_ts_in

        if self.embed_true:
            self.embedding=nn.Linear(self.embedding_dim,d_model)
            self.d_model=d_model
        else:
            self.d_model=self.embedding_dim

        self.positionTensor=positional.myPositionalEncoding(pe_features=self.pe_features,seq_length=seq_length)

        self.encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model,nhead=1,dropout=0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.finalLayer= nn.Linear(self.d_model,num_ts_out)
        self.init_weights()

        #does this have parameters?
        #print(self.positionTensor.parameters())

    def init_weights(self):
        initrange = 0.5
        if self.embed_true:
            self.embedding.bias.data.zero_()
            self.embedding.weight.data.uniform_(-initrange, initrange)
        self.finalLayer.bias.data.zero_()
        self.finalLayer.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    #src input should be seq_length by num_ts
    def forward(self, src, mask: None):
        if self.peconcat_true:
            src=self.positionTensor.concat(src)
            if self.embed_true:
                src=self.embedding(src)
        else:
            if self.embed_true:
                src=self.embedding(src)
            src=self.positionTensor.add(src)

        #print(f"src.size()={src.size()}")
        #print(f"len(src)={len(src)}")
        #batch_size=1 for now
        src_with_batch=src.reshape(self.seq_length,1,self.d_model)

        if mask==None:
            mask = self._generate_square_subsequent_mask(len(src))
        output = self.encoder(src=src_with_batch)
        output = self.finalLayer(output)
        return output







#Both models below are subsumed as special cases of the one above by providing the one above some bool inputs

#the first quick model with only encoder, no embedding
class myEncoderOnly(nn.Module):
    def __init__(self, d_model, num_layers: int=4):
        super().__init__()

        #self.positional_encoder=myPositionalEncoder(num_features)

        self.encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=1,dropout=0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.finalLayer= nn.Linear(d_model,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.finalLayer.bias.data.zero_()
        self.finalLayer.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,src):
        mask = self._generate_square_subsequent_mask(len(src))
        output = self.encoder(src=src,mask=mask)
        output = self.finalLayer(output)
        return output

#the second quick model with only encoder, with embedding
class myEncoderOnlyWithEmbedding(nn.Module):
    def __init__(self, d_model, num_layers: int=4, embedding_dim: int=11):
        super().__init__()

        self.embedding_dim=embedding_dim

        #self.positional_encoder=myPositionalEncoder(num_features)

        self.embedding=nn.Linear(self.embedding_dim,d_model)
        self.encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=1,dropout=0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.finalLayer= nn.Linear(d_model,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5    
        self.embedding.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.finalLayer.bias.data.zero_()
        self.finalLayer.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,src):
        src=self.embedding(src)
        mask = self._generate_square_subsequent_mask(len(src))
        output = self.encoder(src=src,mask=mask)
        output = self.finalLayer(output)
        return output