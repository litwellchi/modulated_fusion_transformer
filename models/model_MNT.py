import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.fc import MLP, FC

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x, ab=None):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        if ab is not None:
            cond = torch.chunk(ab, 2, dim=-1)
            g = cond[0]
            b = cond[1]
            return (self.a_2 + torch.unsqueeze(g, 1)) * (x - mean) / (std + self.eps) + (self.b_2 + torch.unsqueeze(b, 1))
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# ------------------------------------
# ---------- Masking sequence --------
# ------------------------------------
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)



class AttFlat(nn.Module):
    def __init__(self, args):
        super(AttFlat, self).__init__()
        self.args = args
        self.flat_glimpse = 1

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=self.flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            args.hidden_size * self.flat_glimpse,
            args.hidden_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted



class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


class SAG(nn.Module):
    def __init__(self, args):
        super(SAG, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask, cond):
        cond = torch.chunk(cond, 2, dim=-1)
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ), cond[0])

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ), cond[1])

        return y


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Model_MNT(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb):
        super(Model_MNT, self).__init__()
        self.args = args

        # LSTM
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.word_embed_size
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.input_drop = nn.Dropout(args.dropout_i)

        self.lstm_x = nn.LSTM(
            input_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.lstm_y = nn.LSTM(
            input_size=args.audio_feat_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # self.adapter = nn.Linear(args.audio_feat_size, args.hidden_size)
        self.fc = FC(args.hidden_size, args.hidden_size * args.layer * 2 * 2)
        self.enc_list = nn.ModuleList([SA(args) for _ in range(args.layer)])
        self.dec_list = nn.ModuleList([SAG(args) for _ in range(args.layer)])

        # Flatten to vector
        self.attflat_img = AttFlat(args)
        self.attflat_lang = AttFlat(args)

        # Classification layers

        self.proj_norm = LayerNorm(args.hidden_size)
        self.proj = nn.Linear(args.hidden_size, args.ans_size)
        self.proj_drop = nn.Dropout(args.dropout_o)

    def forward(self, x, y, _, flag):
        if flag == 0:
            x_mask = make_mask(x.unsqueeze(2))
            y_mask = make_mask(y.unsqueeze(2))

            embedding = self.embedding(x)

            x, _ = self.lstm_x(self.input_drop(embedding))
            y, _ = self.lstm_x(self.input_drop(embedding))
        elif flag == 1:
            x_mask = make_mask(x)
            y_mask = make_mask(y)

            x, _ = self.lstm_y(self.input_drop(y))
            y, _ = self.lstm_y(self.input_drop(y))
        else:
            x_mask = make_mask(x.unsqueeze(2))
            y_mask = make_mask(y)

            embedding = self.embedding(x)

            x, _ = self.lstm_x(self.input_drop(embedding))
            y, _ = self.lstm_y(self.input_drop(y))

        # Backbone Framework
        for enc in self.enc_list:
            x = enc(x, x_mask)

        lang_feat = self.attflat_lang(
            x,
            x_mask
        )

        cond = torch.chunk(self.fc(lang_feat), self.args.layer, dim=-1)

        for i, dec in enumerate(self.dec_list):
            y = dec(y, y_mask, cond[i])

        img_feat = self.attflat_img(
            y,
            y_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)

        proj_feat = self.proj_drop(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat