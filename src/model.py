import torch
import torch.nn as nn
from einops import rearrange




class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

        self.encoding = self.encoding.unsqueeze(0) 


    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)
    






class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head):
        super(MultiHeadAttention, self).__init__()

        self.dim = dim
        self.n_head = n_head
        self.head_dim = self.dim // self.n_head

        assert self.head_dim * self.n_head == self.dim, "embed_dim must be divisible by num_heads"

        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)

        self.fc_out = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_head)  
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_head)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_head)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1) 

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h s d -> b s (h d)')  

        out = self.fc_out(out)  
        return out



class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size, n_head, ff_hidden_mult=4, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.attn = MultiHeadAttention(dim=embed_size, n_head=n_head)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * ff_hidden_mult),
            nn.ReLU(),
            nn.Linear(embed_size * ff_hidden_mult, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size, n_head, ff_hidden_mult=4, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()

        self.self_attn = MultiHeadAttention(dim=embed_size, n_head=n_head)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = MultiHeadAttention(dim=embed_size, n_head=n_head)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * ff_hidden_mult),
            nn.ReLU(),
            nn.Linear(embed_size * ff_hidden_mult, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # in self attention the target mask is used, to ensure the model
        # does not cheet and considers only valid positions and perceiding tokens
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))


        # in cross attention we use source mask
        # to ensure generated tokens align with input context
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x




class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_size=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        n_head=8,
        max_len=512,
        ff_hidden_mult=4,
        dropout=0.1,
        tokenizer_pad_token_id=0
    ):
        super(TransformerSeq2Seq, self).__init__()

        self.src_embed = nn.Embedding(src_vocab_size, embed_size, padding_idx=tokenizer_pad_token_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=tokenizer_pad_token_id)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        self.pos_decoder = PositionalEncoding(embed_size, max_len)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_size, n_head, ff_hidden_mult, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_size, n_head, ff_hidden_mult, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.tokenizer_pad_token_id = tokenizer_pad_token_id

    def make_src_mask(self, src):
        src_mask = (src != self.tokenizer_pad_token_id).unsqueeze(1).unsqueeze(2) 
        return src_mask 

    def make_tgt_mask(self, tgt):
        tgt_seq_len = tgt.size(1)
        tgt_mask = (tgt != self.tokenizer_pad_token_id).unsqueeze(1).unsqueeze(2)  

        subsequent_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool()  
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)  

        tgt_mask = tgt_mask & subsequent_mask  
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)  
        tgt_mask = self.make_tgt_mask(tgt)  

        enc_out = self.src_embed(src)  
        enc_out = self.pos_encoder(enc_out)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        dec_out = self.tgt_embed(tgt)  
        dec_out = self.pos_decoder(dec_out)
     
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        output = self.fc_out(dec_out)  
        return output




class RewardModel(nn.Module):
    def __init__(self,
                 voc_size,
                 emb_size,
                 hidden_size,
                 n_layers,
                 pad_token_id = 0):
        super(RewardModel, self).__init__()

        self.embedding = nn.Embedding(voc_size, emb_size, padding_idx = pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(d_model = emb_size, nhead = 8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = n_layers)
        self.fc_out = nn.Linear(emb_size, 1)

    def forward(self, seq):
        emb = self.embedding(seq)
        emb = emb.transpose(0, 1)
        output = self.transformer_encoder(emb)
        output = output.mean(dim = 0)
        logits = self.fc_out(output)
        return logits.squeeze()

        

