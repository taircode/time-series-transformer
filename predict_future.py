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