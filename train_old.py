
import torch
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from positional import myPositionalEncoding
import model
import time
import math
import copy
from predict_future import predict_future
from predict_future import predict_future_transformer
import random
import data

def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def mean_normalize(seq: Tensor):
    mean=torch.mean(seq)
    max=torch.max(seq)
    min=torch.min(seq)
    seq_range=max-min
    return (seq-mean)/seq_range, seq_range, mean

#right now prediction_window is hard-coded to 1, but may want to change this later
def create_srctgt_pairs(data, window_size):
    srctgt_data_pairs=[]
    print(f"len(data)={len(data)}")
    L=len(data)
    for i in range(L-window_size-1):
        src_seq=data[i:i+window_size,:]
        tgt_seq=data[i+1:i+1+window_size,:]
        srctgt_data_pairs.append([src_seq,tgt_seq])
    return srctgt_data_pairs
    #note we are returning a list of torch tensors (not one big tensor)

#of the mask_percent indices, 80% are masked, 10% are random, 10% are left alone
def create_bert_tuples(progression,window_size,mask_percent):
    data_tuples=[]
    L=len(progression)
    for i in range(L-window_size-1):
        src_seq=progression[i:i+window_size]
        minimum=torch.min(src_seq)
        maximum=torch.max(src_seq)
        src_seq = torch.FloatTensor(src_seq)
    
        src_seq, seq_range, seq_mean=mean_normalize(src_seq)   
        tgt_seq = copy.deepcopy(src_seq)
        tgt_seq = torch.FloatTensor(tgt_seq) 
        
        num_indices=math.floor(window_size*(mask_percent))

        indices=random.sample(range(window_size),num_indices) #error will be computed only at these positions
        
        num_other=math.floor(num_indices*(.20)) #these are the 10% random + 10% untouched
        indices_other=random.sample(indices,num_other)
        
        indices_mask=[item for item in indices if item not in indices_other] #these are the indices that are masked 

        indices_untouch=random.sample(indices_other,math.floor(num_other/2))
        indices_rand=[item for item in indices_other if item not in indices_untouch]

        for j in indices_rand:
            src_seq[j]=(random.uniform(minimum.item(),maximum.item())-seq_mean)/seq_range

        data_tuples.append([src_seq,tgt_seq,indices,indices_mask,seq_range,seq_mean])
    #srctgt_data_pairs=np.array(srctgt_data_pairs)
    #if you don't do the line directly above, then it's slow to convert a list of np arrays to torch tensors
    return data_tuples

#This method is if you're using the full transformer model with encoder & decoder
#create a lits of data triples (src,tgt,out)
def create_data_triples(progression, window_size, prediction_size):
    data_triples=[]#src, tgt, out
    L=len(progression)
    for i in range(L-window_size-1):
        src_seq=progression[i:i+window_size]
        tgt_seq=progression[i+window_size:i+window_size+prediction_size]
        out_seq=progression[i+window_size+1:i+window_size+prediction_size+1]
        data_triples.append([src_seq,tgt_seq,out_seq])
    #data_triples=np.array(data_triples)
    #if you don't do the line directly above, then it's slow to convert a list of np arrays to torch tensors
    return data_triples
    #return torch.FloatTensor(data_triples)

def get_data(data):
    L=len(data)
    ratio=math.floor(3/4*L)
    train_data=data
    val_data=data[ratio:,]

    if electra:
        #the data tuples are the exactly the same as in bert
        train_data_pairs=create_bert_tuples(train_data,seq_length, .15)
        val_data_pairs=create_bert_tuples(train_data,seq_length, .15)
    elif full_transformer:
        train_data_pairs=create_data_triples(train_data, seq_length, prediction_size)
        val_data_pairs=create_data_triples(val_data, seq_length, prediction_size)
    else: #encoder only
        if train_bert: #BERT-masking
            #hard-coding 15% masking right now
            train_data_pairs=create_bert_tuples(train_data, seq_length, .15)
            val_data_pairs = create_bert_tuples(train_data, seq_length, .15)
        else: #generative, so not BERT-masking
            train_data_pairs=create_srctgt_pairs(train_data, seq_length)
            val_data_pairs=create_srctgt_pairs(val_data, seq_length)
    
    return train_data_pairs, val_data_pairs    

def train_generative(train_data, model):
    model.train()
    print("in train")
    #not doing batching yet but you should
    for i in range(len(train_data)):
        src = train_data[i][0]
        tgt=train_data[i][1]

        #src=src.unsqueeze(1)
        #tgt=tgt.unsqueeze(1)

        src, seq_range, mean=mean_normalize(seq=src)

        if triangle_encoder_mask:
            mask=_generate_square_subsequent_mask(len(src))
            prediction = model(src,mask)
        else:
            prediction = model(src,mask=None)

        #print(f"(before rescale) prediction[-1,]={prediction[-1,]}")
        #print(f"mean={mean}")
        #print(f"seq_range={seq_range}")
        #prediction=prediction*seq_range+mean
        prediction=prediction.view(-1,num_ts_out)
        #if i%1000==0:
        #    print(f"prediction={prediction}")

        #print(f"tgt[-1,]={tgt[-1,]}")

        tgt=(tgt-mean)/seq_range

        if error_last_only:
            loss=criterion(prediction[-1,],tgt[-1,])
        else:
            loss = criterion(prediction, tgt)
        if i % 200 ==0:
            print(f"loss={loss}")

        #unscaled_prediction=prediction*seq_range+mean
        #print(f"prd={unscaled_prediction[-1,]}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss.detach().item())
        #train_loss += loss.detach().item()
        #if loss < .04:
        #    print(f"finished on loop number {i}")
        #    break

def val(val_data, model):
    total_loss=0
    with torch.no_grad():
        for i in range(len(val_data)):
            src = val_data[i][0]
            tgt = val_data[i][1]

            #src=src.unsqueeze(1)
            #tgt=tgt.unsqueeze(1)

            src, seq_range, mean=mean_normalize(seq=src)
            prediction = model(src, mask=None)
            #prediction=prediction*seq_range+mean
            prediction=prediction.view(-1,num_ts_out)

            tgt=(tgt-mean)/seq_range

            loss = criterion(prediction[-1,], tgt[-1,])
            total_loss+=loss
    return total_loss

def create_bert_mask(indices):
    mask = torch.zeros(seq_length,seq_length)
    column=torch.ones((seq_length,1))
    mask[:,indices]=column
    mask = mask.float().masked_fill(mask==1, float('-inf'))
    return mask

def train_bert_style(train_data, model):
    model.train()
    print("in bert train")
    #not doing batching yet but you should
    for i in range(len(train_data)):
        src = train_data[i][0]
        tgt = train_data[i][1]
        error_indices=train_data[i][2]
        mask_indices=train_data[i][3]
        seq_range=train_data[i][4]
        mean=train_data[i][5]

        mask=create_bert_mask(mask_indices)

        prediction = model(src, mask)
        #prediction=prediction*seq_range+mean
        prediction=prediction.view(-1,1)
        #if i%1000==0:
        #    print(f"prediction={prediction}")

        #tgt=(tgt-mean)/seq_range #tgt is already normalized       

        prediction=prediction[error_indices,:]
        tgt=tgt[error_indices,:]

        loss = criterion(prediction, tgt)
        if i % 100 ==0:
            #print(j)
            print(f"loss={loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def bert_val(val_data, model):
    total_loss=0
    with torch.no_grad():
        for i in range(len(val_data)):
            src = val_data[i][0]
            tgt = val_data[i][1]
            error_indices=val_data[i][2]
            mask_indices=val_data[i][3]
            seq_range=val_data[i][4]
            mean=val_data[i][5]

            mask=create_bert_mask(mask_indices)

            prediction = model(src, mask)
            #prediction=prediction*seq_range+mean
            prediction=prediction.view(-1,1)
            #if i%1000==0:
            #    print(f"prediction={prediction}")

            #tgt=(tgt-mean)/seq_range

            prediction=prediction[error_indices,:]
            tgt=tgt[error_indices,:]

            loss = criterion(prediction, tgt)
            #print(f"val_loss={loss}")
            total_loss+=loss
    return total_loss

def mean_normalize_transformer(seq1: Tensor, seq2: Tensor):
    both=torch.cat((seq1,seq2))
    mean=torch.mean(both)
    max=torch.max(both)
    min=torch.min(both)
    seq_range=max-min
    return (seq1-mean)/seq_range, (seq2-mean)/seq_range, seq_range, mean

def train_full_transformer(train_data, model):
    model.train()
    print("in train")
    #not doing batching yet but you should
    for i in range(len(train_data)):
        src = train_data[i][0]
        tgt=train_data[i][1]
        out=train_data[i][2]

        #src=src.unsqueeze(1)
        #tgt=tgt.unsqueeze(1)
        #out=out.unsqueeze(1)

        src, tgt, seq_range, mean=mean_normalize_transformer(seq1=src,seq2=tgt)
        prediction = model(src, tgt)
        #prediction=prediction*seq_range+mean
        prediction=prediction.view(-1,1)
        #print(f"prediction={prediction}")
        #print(f"out={out}")
        #print(prediction.size())
        #print(out.size())
        #if i%1000==0:
        #    print(f"prediction={prediction}")

        out=(out-mean)/seq_range

        #the get data method might produce data triples where the tgt or out falls of the end of the arithmetic sequence making them a smaller length
        #being lazy here, probably better to make this fix in the get_data method(s)
        if prediction.size()==out.size():
            loss = criterion(prediction, out)
            if i % 100 ==0:
                #print(j)
                print(f"loss={loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.detach().item())
            #train_loss += loss.detach().item()
            #if loss < .04:
            #    print(f"finished on loop number {i}")
            #    break

def transformer_val(val_data, model):
    print("in val")
    total_loss=0
    with torch.no_grad():
        for i in range(len(val_data)):
            src = val_data[i][0]
            tgt = val_data[i][1]
            out = val_data[i][2]

            #print(f"src.size()={src.size()}")
            #src=src.unsqueeze(1)
            #print(f"src.size()={src.size()}")
            #tgt=tgt.unsqueeze(1)
            #out=out.unsqueeze(1)

            src, tgt, seq_range, mean=mean_normalize_transformer(seq1=src,seq2=tgt)
            prediction = model(src,tgt)
            #prediction=prediction*seq_range+mean
            prediction=prediction.view(-1,1)

            out=(out-mean)/seq_range
            
            #the get data method might produce data triples where the tgt or out falls of the end of the arithmetic sequence making them a smaller length
            #being lazy here, probably better to make this fix in the get_data method(s)
            if prediction.size()==out.size():
                loss = criterion(prediction, out)
                total_loss+=loss
    return total_loss

def train_electra(train_data, generator, discriminator):
    generator.train()
    discriminator.train()
    print("in electra train")
    #not doing batching yet but you should
    for i in range(len(train_data)):
        src = train_data[i][0]
        tgt = train_data[i][1]
        error_indices=train_data[i][2]
        mask_indices=train_data[i][3]
        seq_range=train_data[i][4]
        mean=train_data[i][5]

        mask=create_bert_mask(mask_indices)

        prediction = generator(src, mask)
        #prediction=prediction*seq_range+mean
        prediction=prediction.view(-1,1)
        #if i%1000==0:
        #    print(f"prediction={prediction}")

        gen_loss = gen_criterion(prediction[error_indices,:], tgt[error_indices,:])

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        #finished with generator

        #do I need requires grad here?
        prediction_copy=prediction.clone().detach().requires_grad_(True)

        dis_src=copy.deepcopy(tgt)
        dis_tgt=[0]*len(tgt)
        for index in mask_indices:
            dis_src[index]=prediction_copy[index]
            dis_tgt[index]=1

        dis_tgt=torch.LongTensor(dis_tgt)

        dis_out=discriminator(dis_src)
        dis_out=dis_out.view(-1,2)

        dis_loss=dis_criterion(dis_out,dis_tgt)
    
        if i % 100 ==0:
            print(f"gen_loss={gen_loss}, dis_loss={dis_loss}")

        dis_optimizer.zero_grad()
        dis_loss.backward()
        dis_optimizer.step()

def electra_val(val_data, generator, discriminator):
    print("in electra val")
    gen_total_loss=0
    dis_total_loss=0
    with torch.no_grad():
        for i in range(len(val_data)):
            src = val_data[i][0]
            tgt = val_data[i][1]
            error_indices=val_data[i][2]
            mask_indices=val_data[i][3]
            seq_range=val_data[i][4]
            mean=val_data[i][5]

            mask=create_bert_mask(mask_indices)

            prediction = generator(src, mask)
            #prediction=prediction*seq_range+mean
            prediction=prediction.view(-1,1)
            
            #if i%1000==0:
            #    print(f"prediction={prediction}")

            #tgt=(tgt-mean)/seq_range

            dis_src=tgt
            dis_tgt=[0]*len(tgt)
            for index in mask_indices:
                dis_src[index]=prediction[index]
                dis_tgt[index]=1

            dis_tgt=torch.LongTensor(dis_tgt)
            #print(f"dis_src.size()={dis_src.size()}")

            dis_out=discriminator(dis_src)

            dis_out=dis_out.view(-1,2)

            #print(f"dis_out={dis_out}")
            #print(f"dis_tgt={dis_tgt}")
            #print(f"dis_out.size()={dis_out.size()}")
            #print(f"dis_tgt.size()={dis_tgt.size()}")
            dis_loss=dis_criterion(dis_out,dis_tgt)
            dis_total_loss+=dis_loss

            prediction=prediction[error_indices,:]
            tgt=tgt[error_indices,:]

            gen_loss = gen_criterion(prediction, tgt)
            #print(f"val_loss={loss}")
            gen_total_loss+=gen_loss
    return gen_total_loss, dis_total_loss

def train(data, model, discriminator=None):

    print("in train")

    data=data

    #create train and val data
    #get_data handles the cases of full_transformer, encoder bert style, or encoder
    train_data, val_data = get_data(data)

    epochs=1000

    #min_loss=float('inf')
    if electra:
        generator=model
        gen_loss, dis_loss=electra_val(val_data,generator,discriminator)
        min_loss=dis_loss
        print(f"gen_loss={gen_loss}")
        print(f"dis_loss={min_loss}")
    elif full_transformer:
        min_loss=transformer_val(val_data, model)
    else:
        if train_bert:
            min_loss=bert_val(val_data, model)
        else:
            min_loss=val(val_data, model)

    print(f"initial model has loss={min_loss}")

    for epoch in range(1,epochs+1):
        
        epoch_start_time=time.time()

        print(f"epoch={epoch}")
        
        if electra:
            train_electra(train_data,generator,discriminator)
        elif full_transformer:
            train_full_transformer(train_data, model)
        else:    
            if train_bert:
                train_bert_style(train_data, model)
            else:
                train_generative(train_data, model)
        

        epoch_end_time=time.time()

        if (epoch-1) %1==0:
            model.eval()
            if electra:
                generator.eval()
                discriminator.eval()
                gen_loss, dis_loss = electra_val(val_data,generator, discriminator)
                print(f"gen_loss={gen_loss}")
                print(f"min_loss={dis_loss}")
                val_loss=dis_loss
            elif full_transformer:
                val_loss=transformer_val(val_data, model)
            else:
                if train_bert:
                    val_loss=bert_val(val_data, model)
                else:
                    val_loss=val(val_data, model)
            
            print('-' * 90)
            print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.5f}'.format(epoch, (epoch_end_time - epoch_start_time), val_loss))
            print('-' * 90)
            if electra:
                if dis_loss<min_loss:
                    print('-' * 89+"\n"+"updating best_model"+"\n"+'-' * 89)
                    min_loss=val_loss
                    best_discriminator=copy.deepcopy(discriminator)
                    best_generator=copy.deepcopy(generator)
                    torch.save(best_discriminator,discriminator_path)
                    torch.save(best_generator,generator_path)
            else:
                if val_loss < min_loss:
                    print('-' * 89+"\n"+"updating best_model"+"\n"+'-' * 89)
                    min_loss=val_loss
                    best_model=copy.deepcopy(model)
                    torch.save(best_model,path)
                else:
                    if full_transformer:
                        predictions=predict_future_transformer(model,data,seq_length,30,tgt_seq_length)
                    else:
                        predictions=predict_future(model, data, seq_length, 30, num_ts_out)
                    new_predictions=predictions[-30:]
                    print(new_predictions.view(1,-1))

#add num_ts_in, right now it's hardcoded to 1
seq_length=90
#maybe have it figure out num_ts from dataLoader
num_ts_in=1
num_ts_out=1
pe_features=10
#positionTensor = myPositionalEncoding(pe_features=pe_features, seq_length=seq_length)

#if myEncoderOnly then d_model=embedding_dim
#if myEncoderOnlyWithEmbedding then d_model=512, emedding_dim=embedding_dim

from_new=True

electra=False

#can't both be true
train_bert=True
full_transformer=False

#options for encoder-only
error_last_only=True
if error_last_only:
    triangle_encoder_mask=False
else:
    triangle_encoder_mask=True

embed_true=False
peconcat_true=True

dtype=torch.float

if full_transformer:
    type="transformer"
elif train_bert:
    type="bert"
else:
    if error_last_only:
        type="encoder_error_last"
    else:
        type="encoder_error_all"

if embed_true:
    firstlayer="embed"
else:
    firstlayer="noembed"
if peconcat_true:
    positional="concat"
else:
    positional="add"

if electra:
    generator_path="electra/generator.pth"
    discriminator_path="electra/discriminator.pth"
else:   
    path=type+"/"+firstlayer+"/"+positional+"/model.pth"

tgt_seq_length=2
prediction_size=tgt_seq_length

if from_new:
    if electra:
        generator = model.myEncoder(d_model=8, num_ts_in=num_ts_in, num_ts_out=num_ts_out, seq_length=seq_length, pe_features=pe_features, embed_true=embed_true,peconcat_true=peconcat_true)
        discriminator = model.myDiscriminator(d_model=16, num_layers=4, seq_length=seq_length, num_ts_in=num_ts_out)
    elif full_transformer:
        mymodel = model.myTransformer(d_model=8, 
        nhead=1, 
        input_layer_true=embed_true, 
        peconcat_true=peconcat_true, 
        num_ts=1, 
        src_seq_length=seq_length, 
        tgt_seq_length=tgt_seq_length,
        num_encoder_layers=4,
        num_decoder_layers=4,
        pe_features=10) #pe_features only matters if peconcat_true=True
    else:
        mymodel = model.myEncoder(d_model=16, num_ts_in=num_ts_in, num_ts_out=num_ts_out, seq_length=seq_length, pe_features=pe_features, embed_true=embed_true,peconcat_true=peconcat_true)
else:
    if electra:
        generator=torch.load(generator_path)
        discriminator=torch.load(discriminator_path)
    else:
        mymodel=torch.load(path)

dataLoader=data.myDataLoader()
data=dataLoader.get_data()

if electra:
    print("implement electra prediction")
elif full_transformer:
    predictions=predict_future_transformer(mymodel, data, seq_length, 90, tgt_seq_length)
    new_predictions=predictions[-90:]
    print(new_predictions.view(1,-1))
else:
    predictions=predict_future(mymodel, data, seq_length, 90, num_ts_out)
    new_predictions=predictions[-90:]
    print(new_predictions.view(1,-1))

if electra:
    optim_lr= 1e-8
    dis_lr=1e-3
    gen_criterion = torch.nn.MSELoss()
    gen_optimizer = torch.optim.SGD(generator.parameters(), lr=optim_lr)
    dis_criterion = torch.nn.CrossEntropyLoss()
    dis_optimizer =torch.optim.Adam(discriminator.parameters(), lr=dis_lr) 
else:
    learning_rate = 1e-8
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=learning_rate)

print(f"electra={electra}")
print(f"full_transformer={full_transformer}")
print(f"Bert={train_bert}")
print(f"embed_true={embed_true}")
print(f"peconcat_true={peconcat_true}")
if electra:
    train(data,generator,discriminator)
else:
    train(data, mymodel)

#PUT IN ELECTRA CASES DOWN HERE
if electra:
    generator=torch.load(generator_path)
    discriminator=torch.load(discriminator_path)
else:
    mymodel=torch.load(path)

if electra:
    print("implement electra prediction")
elif full_transformer:
    predictions=predict_future_transformer(mymodel, data, seq_length, 90, tgt_seq_length)
else:
    predictions=predict_future(mymodel, data, seq_length, 90, num_ts_out)
new_predictions=predictions[-90:]
print(new_predictions.view(1,-1))