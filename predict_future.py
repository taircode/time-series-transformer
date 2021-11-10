import torch
from torch import Tensor

def mean_normalize(seq: Tensor):
    mean=torch.mean(seq)
    max=torch.max(seq)
    min=torch.min(seq)
    seq_range=max-min
    return (seq-mean)/seq_range, seq_range, mean

def predict_future(eval_model, progression: Tensor, src_length, num_steps, num_ts_out):
    eval_model.eval()

    src_seq=progression

    with torch.no_grad():
        for i in range(num_steps):
            #print(f"src_seq={src_seq[-src_length:]}")
            input=src_seq[-src_length:]
            #print(input)
            #print(f"input.size()={input.size()}")
            #print(f"input={input}")
            input, seq_range, mean=mean_normalize(seq=input)
            output=eval_model(src=input,mask=None)
            output=output*seq_range+mean
            #print(f"output={output[-1:]}")
            #print(f"output.size()={output.size()}")
            #print(f"output={output}")
            output=output.view(-1,num_ts_out)
            #print(f"output={output}")
            #print(f"output.size()={output.size()}")
            #print(f"output={output}")
            #print(f"new output={output[-1:]}")
            src_seq=torch.cat((src_seq, output[-1:]))
    
    return src_seq

def mean_normalize_transformer(seq1: Tensor, seq2: Tensor):
    both=torch.cat((seq1,seq2))
    mean=torch.mean(both)
    max=torch.max(both)
    min=torch.min(both)
    seq_range=max-min
    return (seq1-mean)/seq_range, (seq2-mean)/seq_range, seq_range, mean

def predict_future_transformer(eval_model, progression: Tensor, src_length, num_steps, tgt_length):
    eval_model.eval()

    src_seq=progression

    with torch.no_grad():
        for i in range(num_steps):
            input=src_seq[-src_length-(tgt_length-1):-(tgt_length-1)]
            tgt=src_seq[-tgt_length:]
            #print(f"input.size()={input.size()}")
            #print(f"input={input}")
            input, tgt, seq_range, mean=mean_normalize_transformer(seq1=input,seq2=tgt)
            output=eval_model(input,tgt)
            output=output*seq_range+mean
            #print(f"output.size()={output.size()}")
            #print(f"output={output}")
            output=output.view(-1,1)
            #print(f"output.size()={output.size()}")
            #print(f"output={output}")
            src_seq=torch.cat((src_seq, output[-1:]))
    return src_seq


#seq_length=100
#best_model=torch.load("best_model.pth")
#dtype=torch.float

#progression=torch.arange(start=3500,end=4000,step=3,dtype=dtype)
#progression=progression.unsqueeze(1)
#print(f"progression.size()={progression.size()}")
#print(f"progression={progression}")

#print(f"input={progression.view(1,-1)}")

#predictions=predict_future(best_model, progression, 10)

#new_predictions=predictions[-10:]
#print(f"new_predictions.size()={new_predictions.size()}")

#actual=torch.arange(start=3500,end=4029,step=3,dtype=dtype)
#actual=actual.unsqueeze(1)
#actual=actual[-10:]
#print(f"actual.size()={actual.size()}")


#dif=torch.add(new_predictions,actual,alpha=-1)
#print(f"dif.size()={dif.size()}")


#print(f"actual={actual.view(1,-1)}")
#print(f"predictions={new_predictions.view(1,-1)}")
#print(f"dif={dif.view(1,-1)}")