
from typing import Literal, Union
import torch
import torch.nn.utils.rnn as rnn_utils
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

def pad_sequences(sequences: Union[list[torch.Tensor], rnn_utils.PackedSequence],
                  padding_value: int = 0,
                  padding_side : Literal['left', 'right'] = 'right'):
    '''
    Args:
        sequences: (seq_len, *) 
    
    Returns:
        (batch_size, max(seq_len), *), (batch_size, )
    '''
    if isinstance(sequences, rnn_utils.PackedSequence):
        sequences = rnn_utils.unpack_sequence(sequences)

    padded_sequences = rnn_utils.pad_sequence(sequences, 
                                              batch_first = True,
                                              padding_value = padding_value,
                                              padding_side = padding_side
                                              )
    
    lengths = torch.tensor([k.size(0) for k in sequences]).long().to(padded_sequences.device)
    
    return padded_sequences, lengths

def unpad_sequences(padded_sequences: torch.Tensor, lengths: torch.Tensor,
                    padding_side: Literal['left', 'right'] = 'right'):
    if padding_side == 'left':
        padded_sequences = padded_sequences.flip(dims = [1])
    
    sequences = rnn_utils.unpad_sequence(padded_sequences = padded_sequences,
                                         lengths = lengths,
                                         batch_first = True)
    
    if padding_side == 'left':
        return [k.flip(dims = [0]) for k in sequences]
    
    return sequences



def pack_sequences(sequences: list[torch.Tensor]):
    packed_sequences = torch.concat(sequences)

    cumsum_len = torch.tensor([0] + [s.size(0) for s in sequences],
                              dtype = torch.int32,
                              device = packed_sequences.device).cumsum(0)
    
    return packed_sequences, cumsum_len



def unpack_sequences(packed_sequences: torch.Tensor, cumsum_len: torch.Tensor):
    return [packed_sequences[l: r] for l, r in zip(cumsum_len[: -1], cumsum_len[1: ])]






def mask2position(attention_mask: torch.Tensor):
    '''attention_mask: (total_seq_len,)  [[1, 1, ..., 4, 4, 4, 5, 5, 0, 0]]'''
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids



# def position2mask(position_ids: torch.Tensor):
#     '''position_ids: (total_seq_len,)  [[0, 1, ..., 0, 1, 2, 0, 1, 0, 0]]'''









    # def forward(self,x, xl):
        # b,t,c,h,w=x.shape
        # x=rearrange(x,"b t c h w -> b c t h w")
        # x=self.front3D(x)
        # x=rearrange(x,"b c t h w -> (b t) c h w")
        # x=self.resnet(x)
        
        # x=self.fc(x)

        # return rearrange(x,"(b t) f -> b t f",b=b)

        
        # x=rearrange(x,"b t c h w -> b c t h w")
        # x=self.front3D(x)
        # x=rearrange(x,"b c t h w -> b t c h w")
        
        
        # # (b, t, c, h, w)  ->  (bt, c, h, w)
        # packed_x=rnn_utils.pack_padded_sequence(x.transpose(0,1),xl.cpu(),enforce_sorted=False)

        # x=self.resnet(packed_x.data)

        # #(bt, f)
        # x=self.fc(x)

        # packed_x =rnn_utils.PackedSequence(data = x,
        #                                   batch_sizes = packed_x.batch_sizes,
        #                                   sorted_indices = packed_x.sorted_indices
        #                                   )
        # padded_x, pxl= rnn_utils.pad_packed_sequence(packed_x)
        # return padded_x.transpose(0,1)