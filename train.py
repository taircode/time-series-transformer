
import torch
from torch import Tensor
import numpy as np
from positional import myPositionalEncoding
import model
import time
import math
import copy
from predict_future import predict_future
import random

import data

#add num_ts_in, right now it's hardcoded to 1
seq_length=90
#maybe have it figure out num_ts from dataLoader
num_ts_in=1
num_ts_out=1
pe_features=10
positionTensor = myPositionalEncoding(pe_features=pe_features, seq_length=seq_length)

#if myEncoderOnly then d_model=embedding_dim
#if myEncoderOnlyWithEmbedding then d_model=512, emedding_dim=embedding_dim

train_bert=False

embed_true=True
peconcat_true=True

model = model.myEncoder(d_model=16, num_ts_in=num_ts_in, num_ts_out=num_ts_out, seq_length=seq_length, pe_features=pe_features, embed_true=embed_true,peconcat_true=peconcat_true)
criterion = torch.nn.MSELoss()

learning_rate = 1e-8
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

dtype=torch.float

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

def get_data(data):
    L=len(data)
    ratio=math.floor(3/4*L)
    train_data=data
    val_data=data[ratio:,]

    train_data_pairs=create_srctgt_pairs(train_data, seq_length)
    val_data_pairs=create_srctgt_pairs(val_data, seq_length)
    
    return train_data_pairs, val_data_pairs

def train_predictive(train_data):
    model.train()
    print("in train")
    #not doing batching yet but you should
    for i in range(len(train_data)):
        src = train_data[i][0]
        tgt=train_data[i][1]

        #src=src.unsqueeze(1)
        #tgt=tgt.unsqueeze(1)

        src, seq_range, mean=mean_normalize(seq=src)
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

        loss = criterion(prediction, tgt)
        if i % 100 ==0:
            #print(j)
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

def val(val_data):
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

            loss = criterion(prediction, tgt)
            total_loss+=loss
    return total_loss

def create_bert_tuples(progression,window_size,mask_percent):
    data_tuples=[]
    L=len(progression)
    for i in range(L-window_size-1):
        src_seq=progression[i:i+window_size]
        minimum=min(src_seq)
        maximum=max(src_seq)
        src_seq = torch.FloatTensor(src_seq)
    
        src_seq, seq_range, seq_mean=mean_normalize(src_seq)   
        tgt_seq=copy.deepcopy(src_seq)
        tgt_seq = torch.FloatTensor(tgt_seq) 
        
        num_indices=math.floor(window_size*(mask_percent))

        indices=random.sample(range(window_size),num_indices)
        
        num_other=math.floor(num_indices*(.20))
        indices_other=random.sample(indices,num_other)
        
        indices_mask=[item for item in indices if item not in indices_other]

        indices_untouch=random.sample(indices_other,math.floor(num_other/2))
        indices_rand=[item for item in indices_other if item not in indices_untouch]

        for j in indices_rand:
            src_seq[j]=(random.randint(minimum,maximum)-seq_mean)/seq_range

        data_tuples.append([src_seq,tgt_seq,indices,indices_mask,seq_range,seq_mean])
    #srctgt_data_pairs=np.array(srctgt_data_pairs)
    #if you don't do the line directly above, then it's slow to convert a list of np arrays to torch tensors
    return data_tuples

def get_bert_data():
    train_progression=get_arithmetic_prog(0,6000,arithmetic_step)
    val_progression=get_arithmetic_prog(5000,5600,arithmetic_step)

    #hard coded 15 percent for now
    train_data=create_bert_tuples(train_progression, seq_length, .15)
    val_data=create_bert_tuples(val_progression, seq_length, .15)
    
    return train_data, val_data

def create_bert_mask(indices):
    mask = torch.zeros(seq_length,seq_length)
    column=torch.ones((seq_length,1))
    mask[:,indices]=column
    mask = mask.float().masked_fill(mask==1, float('-inf'))
    return mask

def train_bert_style(train_data):
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

        src=src.unsqueeze(1)
        tgt=tgt.unsqueeze(1)

        prediction = model(src, mask)
        prediction=prediction*seq_range+mean
        prediction=prediction.view(-1,1)
        #if i%1000==0:
        #    print(f"prediction={prediction}")

        prediction=prediction[error_indices,:]
        tgt=tgt[error_indices,:]

        loss = criterion(prediction, tgt)
        if i % 100 ==0:
            #print(j)
            print(f"loss={loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def bert_val(val_data):
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

            src=src.unsqueeze(1)
            tgt=tgt.unsqueeze(1)

            prediction = model(src, mask)
            prediction=prediction*seq_range+mean
            prediction=prediction.view(-1,1)
            #if i%1000==0:
            #    print(f"prediction={prediction}")

            prediction=prediction[error_indices,:]
            tgt=tgt[error_indices,:]

            loss = criterion(prediction, tgt)
            #print(f"val_loss={loss}")
            total_loss+=loss
    return total_loss


def train(data, model):

    data=data

    #create train and val data
    if train_bert:
        train_data, val_data = get_bert_data()
    else:
        train_data, val_data = get_data(data)

    epochs=50

    min_loss=float('inf')

    best_model=copy.deepcopy(model)

    for epoch in range(1,epochs+1):
        
        epoch_start_time=time.time()

        print(f"epoch={epoch}")
        
        if train_bert:
            train_bert_style(train_data)
        else:
            train_predictive(train_data)
        

        epoch_end_time=time.time()

        if (epoch-1) %1==0:
            model.eval()
            if train_bert:
                val_loss=bert_val(val_data)
            else:
                val_loss=val(val_data)
            
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.5f}'.format(epoch, (epoch_end_time - epoch_start_time), val_loss))
            print('-' * 89)

            if val_loss < min_loss:
                print('-' * 89+"\n"+"updating best_model"+"\n"+'-' * 89)
                min_loss=val_loss
                best_model=copy.deepcopy(model)
                torch.save(best_model,"best_model.pth")

            #src_sequence=torch.arange(3500,3600,dtype=dtype)
            #src_sequence=src_sequence.unsqueeze(1)
            #src_sequence, seq_range, mean=mean_normalize(seq=src_sequence)
            #out=model(src_sequence)
            #out=out*seq_range+mean
            #out=out.unsqueeze(1)
            #print(out)

        #print(f"train_loss={train_loss}")
        #if train_loss<100:
        #    break


dataLoader=data.myDataLoader()
data=dataLoader.get_data()

mymodel=torch.load("best_model.pth")

#train(data)
train(data, mymodel)

predictions=predict_future(mymodel, data, seq_length, 10, num_ts_out)
new_predictions=predictions[-10:]
print(new_predictions)


#progression=torch.arange(start=3500,end=4000,step=arithmetic_step,dtype=dtype)
#progression=progression.unsqueeze(1)

#predictions=predict_future(best_model, progression, 10)

#print(f"input={progression.view(1,-1)}")

#new_predictions=predictions[-10:]

#actual=torch.arange(start=3500,end=4029,step=arithmetic_step,dtype=dtype)
#actual=actual.unsqueeze(1)
#actual=actual[-10:]
#
#dif=torch.add(new_predictions,actual,alpha=-1)

#print(f"actual={actual.view(1,-1)}")
#print(f"predictions={new_predictions.view(1,-1)}")
#print(f"dif={dif.view(1,-1)}")