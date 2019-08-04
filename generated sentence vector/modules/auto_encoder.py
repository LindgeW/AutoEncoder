import torch
import torch.nn as nn
from .rnn_encoder import RNNEncoder


# 自编码器
class AutoEncoder(nn.Module):
    def __init__(self, args, embedding_weights):
        super(AutoEncoder, self).__init__()
        self._args = args

        embed_dim = embedding_weights.shape[1]
        self._wd_embed = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights))

        self._bidirectional = True
        nb_directions = 2 if self._bidirectional else 1

        self.encoder = RNNEncoder(input_size=embed_dim,
                                  hidden_size=args.hidden_size,
                                  num_layers=args.num_layers,
                                  bidirectional=self._bidirectional,
                                  batch_first=True,
                                  rnn_type='lstm')

        self.decoder = RNNEncoder(input_size=nb_directions * args.hidden_size,
                                  hidden_size=embed_dim // nb_directions,
                                  num_layers=args.num_layers,
                                  bidirectional=self._bidirectional,
                                  batch_first=True,
                                  rnn_type='lstm')

        # self._hidden2tag = nn.Linear(args.hidden_size * nb_directions, args.tag_size)

        self._embed_dropout = nn.Dropout(args.embed_dropout)
        self._linear_dropout = nn.Dropout(args.linear_dropout)

    def forward(self, wd_idxs, non_pad_mask=None):
        '''
        :param wd_idxs: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len)  非填充部分mask
        :return:
        '''
        seq_len = wd_idxs.size(1)

        if non_pad_mask is None:
            # 非取mask
            non_pad_mask = wd_idxs.ne(self._args.pad)

        # (bz, seq_len, embed_dim)
        wd_embed = self._wd_embed(wd_idxs)

        if self.training:
            embed = self._embed_dropout(wd_embed)
        else:
            embed = wd_embed

        # (num_layers, batch_size, nb_directions * hidden_size)
        # 编码器输出词序列的句向量
        _, hidden_n = self.encoder(embed, non_pad_mask)
        # (batch_size, nb_directions * hidden_size)
        # sent_vec = hidden_n[0][-1]  # 句向量

        # (batch_size, nb_directions * hidden_size) ->
        # (batch_size, seq_len, nb_directions * hidden_size)
        sent_enc = hidden_n[0][-1].unsqueeze(1).repeat(1, seq_len, 1)

        if self.training:
            sent_enc = self._linear_dropout(sent_enc)

        # (batch_size, seq_len, embed_dim)
        dec_out, _ = self.decoder(sent_enc, non_pad_mask)

        # 用于计算编码MSE误差(无监督)
        return dec_out, wd_embed
