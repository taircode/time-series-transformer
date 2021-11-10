import torch
import torch.nn as nn
import positional

#A model with only an encoder
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

    #def _generate_square_subsequent_mask(self, sz):
    #    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #    return mask

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
            output = self.encoder(src=src_with_batch)
        else:
            output = self.encoder(src=src_with_batch,mask=mask)
        output = self.finalLayer(output)
        return output

#A model with an encoder and decoder
class myTransformer(nn.Module):
        def __init__(self, 
        d_model: int=512, 
        nhead: int=1, 
        input_layer_true: bool=True, 
        peconcat_true: bool=False, 
        num_ts: int=1, 
        src_seq_length: int=100, 
        tgt_seq_length: int=2,
        num_encoder_layers: int=4,
        num_decoder_layers: int=4,
        pe_features: int=10
        ):
            super().__init__()

            self.num_ts=num_ts
            self.peconcat_true=peconcat_true
            self.input_layer_true=input_layer_true
            self.src_seq_length=src_seq_length
            self.tgt_seq_length=tgt_seq_length

            if self.peconcat_true:
                self.num_features=num_ts+pe_features
                self.pe_features=pe_features
            else:
                #in this case num_features=num_ts=pe_features
                self.num_features=num_ts
                self.pe_features=d_model

            #this is the default, not sure why you wouldn't embed. If you don't, then d_model=self.num_features
            if self.input_layer_true:
                self.src_input_layer=nn.Linear(self.num_features,d_model)
                self.tgt_input_layer=nn.Linear(self.num_features,d_model)
                self.d_model=d_model
            else:
                self.d_model=self.num_features

            self.srcPositionTensor=positional.myPositionalEncoding(pe_features=self.pe_features,seq_length=src_seq_length)
            self.tgtPositionTensor=positional.myPositionalEncoding(pe_features=self.pe_features,seq_length=tgt_seq_length)

            self.transformer=nn.Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers)
            
            #change 1 to num_ts_out later, for now hard-coded
            self.output_layer=nn.Linear(d_model,1)

            self.init_weights()


        def init_weights(self):
            initrange = 0.5
            self.output_layer.bias.data.zero_()
            self.output_layer.weight.data.uniform_(-initrange, initrange)
            if self.input_layer_true:
                self.src_input_layer.bias.data.zero_()
                self.src_input_layer.weight.data.uniform_(-initrange, initrange)
                self.tgt_input_layer.bias.data.zero_()
                self.tgt_input_layer.weight.data.uniform_(-initrange, initrange)

        def _generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

        #src input should be (seq_length, batch_size, num_ts)
        def forward(self, src, tgt):
            if self.peconcat_true:
                src=self.srcPositionTensor.concat(src)
                tgt=self.tgtPositionTensor.concat(tgt)
                if self.input_layer_true:
                    src=self.src_input_layer(src)
                    tgt=self.tgt_input_layer(tgt)
            else:
                if self.input_layer_true:
                    src=self.src_input_layer(src)
                    tgt=self.tgt_input_layer(tgt)
                src=self.srcPositionTensor.add(src)
                tgt=self.tgtPositionTensor.add(tgt)

            #print(f"src.size()={src.size()}")
            #print(f"len(src)={len(src)}")
            #batch_size=1 for now
            src_with_batch=src.reshape(self.src_seq_length,1,self.d_model)
            tgt_with_batch=tgt.reshape(self.tgt_seq_length,1,self.d_model)

            mask = self._generate_square_subsequent_mask(len(tgt))
            output = self.transformer(src=src_with_batch,tgt=tgt_with_batch,tgt_mask=mask)
            output=self.output_layer(output)
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