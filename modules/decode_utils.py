import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List
from torch import Tensor, zeros, cat as torch_cat
import random 

def greedy_decode(x,encoder,decoder,embeddings,classifier,max_length):
    bs, _, _ = x.size()
    device = x.device
    eos_token = 9
    sos_token = 0

    encoder_output: Tensor = encoder(x)      
    if isinstance(encoder_output,tuple):
        encoder_1, encoder_2 = encoder_output
        encoder_1 = encoder_1.permute(1,0,2)
        encoder_2 = encoder_2.permute(1,0,2)
    else:
        encoder_output = encoder_output.permute(1,0,2)                                                                                                                                           
    decoded_batch = torch.zeros((max_length,bs), device=device).long()

    for t in range(1,max_length):
        h_ = embeddings(decoded_batch[:t])
        
        if isinstance(encoder_output,tuple):
            decoder_out = decoder(
                h_,
                encoder_2,
                encoder_1,
                attention_mask = None
            )
        else:
            decoder_out = decoder(
                h_,
                encoder_output,
                attention_mask = None
            )

        out = classifier(decoder_out)
        out = F.log_softmax(out,dim=-1)        

        topv, topi = out[-1].data.topk(1)
        decoded_batch[t,:] = topi.view(-1)
    return decoded_batch


def topk_sampling(x,encoder,decoder,embeddings,classifier,max_length,p=.5):
    bs, _, _ = x.size()
    device = x.device
    eos_token = 9
    sos_token = 0

    encoder_output: Tensor = encoder(x)      
    encoder_output = encoder_output.permute(1,0,2)                                                                                                                                           
    decoded_batch = torch.zeros((max_length,bs), device=device).long()
    for t in range(1,max_length):
        h_ = embeddings(decoded_batch[:t])
        decoder_out = decoder(
            h_,
            encoder_output,
            attention_mask = None
        )

        out = classifier(decoder_out)
        out = F.softmax(out,dim=-1)        
        idx = torch.argsort(out, descending=True)
        res, cumsum = [], 0
        for i in idx[0][0]:
            res.append(i)
            cumsum += out.view(-1)[i]
            if cumsum > p:
                decoded_batch[t,:] = idx.view(-1).new_tensor([random.choice(res)]) 
                break
    return decoded_batch