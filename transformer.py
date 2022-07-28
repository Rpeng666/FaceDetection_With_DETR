import torch
import torch.nn as nn
import torch.nn.functional as F



class Mlp(nn.Module):  # 标准Transformer中的FFN部分
    def __init__(self, embed_dim, mlp_ratio) -> None:
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    '''多头注意力部分实现'''
    def __init__(self, embed_dim, num_heads) -> None:
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.scales = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, self.all_head_dim)
        self.k = nn.Linear(embed_dim, self.all_head_dim)
        self.v = nn.Linear(embed_dim, self.all_head_dim)

        self.proj = nn.Linear(self.all_head_dim, embed_dim)
        self.dropout = nn.Dropout()
        self.attention_dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim = -1)

    def transpose_multihead(self, x):  # [..., head_dim * num_heads ]
        new_shape = list(x.shape[:-1]) + [self.num_heads, self.head_dim] #[..., num_heads, head_dim]
        x = x.reshape(new_shape) 
        x = x.flatten(1, 2)
        x = x.permute([1, 0, 2])
        return x

    def forward(self, query, key, value):
        key_len = key.shape[0] 
        batch_size = key.shape[1] 
        query_len = query.shape[0] 
        embed_dim = query.shape[2] 
    
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        q, k, v = map(self.transpose_multihead, [q, k, v])

        attn = torch.bmm(q, k.permute(0, 2, 1)) # q * k'

        attn = attn * self.scales
        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute([1, 0, 2])
        out = out.reshape([query_len, batch_size, embed_dim])

        out = self.proj(out)
        out = self.dropout(out)

        return out


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super(Encoder, self).__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        # self.attention = Attention(embed_dim, num_heads)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, 4)

    def forward(self, x, pos_embed):
        h = x   #[h*w, batch_size, channels]
        x = self.layernorm1(x)
        
        q = x + pos_embed  #[h*w , batch_size, embeding_dim ]
        k = x + pos_embed  #[h*w , batch_size, embeding_dim ]

        x = self.attention(q, k, x)[0]

        x = x + h

        h = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = x + h

        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super(Decoder, self).__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        # self.attention1 = Attention(embed_dim, num_heads)
        self.attention1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        # self.attention2 = Attention(embed_dim, num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, 4)

    def forward(self, x, encoder_out, pos, query_pos):
        # x是object_queries
        h = x     # [num_queries, batch, embed_dim]
        x = self.layernorm1(x)
        q = x + query_pos  # [num_queries, batch, embed_dim]
        k = x + query_pos  # [num_queries, batch, embed_dim]

        x = self.attention1(q,k,x)[0]
        x = h + x

        h = x
        x = self.layernorm2(x)
        q = x + query_pos
        k = encoder_out + pos  # [h*w, batch_size, embeding_dim ]
        v = encoder_out        # [h*w, batch_size, embeding_dim ]

        x = self.attention2(q, k, v)[0]
        x = h + x

        h = x
        x = self.layernorm3(x)
        x = self.mlp(x)
        x = x + h

        return x


class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
        self.embed_dim = 256
        self.num_heads = 4
        self.num_encoders = 4
        self.num_decoders = 4
        
        self.encoder = nn.ModuleList([Encoder(self.embed_dim, self.num_heads) for i in range(self.num_encoders)])
        self.decoder = nn.ModuleList([Decoder(self.embed_dim, self.num_heads) for i in range(self.num_decoders)])

        self.encoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, pos_embed, query_embed):
        # query_embed是100*emb_dim的 nn.embedding
        # pos_embed还是feature_map里的位置嵌入，在decoder里又使用一边大概是为了加强位置信息
        batch_size, channels, h, w = x.shape
        x = x.flatten(2) #[batch_size, channels, h*w ]
        x = x.permute([2, 0, 1])  #[h*w, batch_size, channels]

        pos_embed = pos_embed.flatten(2) # pos_embed和x的形状完全相同，对其进行和x一样的变形操作
        pos_embed = pos_embed.permute([2, 0, 1])

        for encoder in self.encoder:
            encoder_out = encoder(x, pos_embed)

        encoder_out = self.encoder_norm(encoder_out)

        query_embed = query_embed.unsqueeze(1) # [num_queries, 1, embed_dim]
        query_embed = query_embed.repeat(1, batch_size, 1) # [num_queries, batch, embed_dim]

        target = torch.zeros_like(query_embed)  #[num_queries, batch, embed_dim]

        for decoder in self.decoder:
            decoder_out = decoder(target, encoder_out, pos_embed, query_embed)
        
        decoder_out = self.decoder_norm(decoder_out)

        return decoder_out


def main():
    # 测试部分
    model = Transformer()
    print(model)

if __name__ == '__main__':
    main()