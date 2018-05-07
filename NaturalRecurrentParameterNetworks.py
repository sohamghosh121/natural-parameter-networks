
import torch
import numpy as np
import torch.nn as nn
import torch.optim as O
import torch.autograd as autograd
import torch.nn.functional as F

from NaturalParameterNetworks import GaussianNPNLinearLayer, GaussianNPNNonLinearity

PI = float(np.pi)
ETA_SQ = float(np.pi / 8.0)
# for sigmoid
ALPHA = float(4.0 - 2.0 * np.sqrt(2))
ALPHA_SQ = float(ALPHA ** 2.0)
BETA = - float(np.log(np.sqrt(2) + 1))
# for tanh
ALPHA_2 = float(8.0 - 4.0 * np.sqrt(2))
ALPHA_2_SQ = float(ALPHA ** 2.0)
BETA_2 = - float(0.5 * np.log(np.sqrt(2) + 1))

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    has_cuda = True
    tt = torch.cuda
else:
    has_cuda = False
    tt = torch


class GaussianNPRN(nn.Module):
    """
        TODO: try to reuse code from NaturalParameterNetworks.py
    """
    def __init__(self, input_features, hidden_sz, eps=0.1, variant='gru'):
        super(GaussianNPRN, self).__init__()
        if variant == 'gru':
            self.Wz_in_m, self.Mz_in_s = self.get_init_W(input_features, hidden_sz)
            self.Wz_h_m, self.Mz_h_s = self.get_init_W(hidden_sz, hidden_sz)
            self.bz_m, self.pz_s = self.get_init_b(hidden_sz)
            self.Wr_in_m, self.Mr_in_s = self.get_init_W(input_features, hidden_sz)
            self.Wr_h_m, self.Mr_h_s = self.get_init_W(hidden_sz, hidden_sz)
            self.br_m, self.pr_s = self.get_init_b(hidden_sz)
            self.Ws_in_m, self.Ms_in_s = self.get_init_W(input_features, hidden_sz)
            self.Ws_h_m, self.Ms_h_s = self.get_init_W(hidden_sz, hidden_sz)
            self.bs_m, self.ps_s = self.get_init_b(hidden_sz)

            self.sigmoid = GaussianNPNNonLinearity('sigmoid')
            self.tanh = GaussianNPNNonLinearity('tanh')

            self.input_features = input_features
            self.hidden_sz = hidden_sz

        else:
            raise NotImplementedError('Only GRU implemented.')

    def linear_layer(self, x, h, W, U, b):
        x_m, x_s = x
        h_m, h_s = h
        W_m, W_s = W
        U_m, U_s = U
        b_m, b_s = b
        o_m = b_m + torch.matmul(x_m, W_m) + torch.matmul(h_m, U_m)
        o_s = b_s + torch.matmul(x_s, W_s) + \
            torch.matmul(x_s, torch.pow(W_m, 2)) + \
            torch.matmul(torch.pow(x_m, 2), W_s) + \
            torch.matmul(h_s, torch.pow(U_s, 2)) + \
            torch.matmul(torch.pow(h_m, 2), U_s)
        return o_m, o_s

    def get_init_W(self, _in, _out):
        # TODO: initialisation probably needs to change?
        W_m = 2 * np.sqrt(6)/ np.sqrt(_in + _out) * (np.random.rand(_in, _out) - 0.5)
        W_s = 1 * np.sqrt(6)/ np.sqrt(_in + _out) * (np.random.rand(_in, _out))
        M_s = np.log(np.exp(W_s) - 1)
        return nn.Parameter(tt.FloatTensor(W_m)), nn.Parameter(tt.FloatTensor(M_s))

    def get_init_b(self, _out):
        b_m = np.zeros((_out))
        # instead of bias_s, parametrize as log(1+exp(pias_s))
        p_s = np.exp(-1 * np.ones((_out)))
        return nn.Parameter(tt.FloatTensor(b_m)), nn.Parameter(tt.FloatTensor(p_s))

    def pos_variance_transform(self, s):
        return torch.log(1 + torch.exp(s))

    def forward(self, input, h_m, h_s):
        # always assume input_s is zero
        if type(input) is tuple: # directly sending input_m, input_s
            input_m, input_s = input
        elif type(input) is torch.Tensor:
            input_m = input
            input_s = torch.zeros(input_m.size())
            if has_cuda:
                input_s = input_s.cuda()
            input_s = autograd.Variable(input_s)
        else:
            raise ValueError('Got %s for input' % type(input))

        # do this to ensure positivity
        Wz_in_s = self.pos_variance_transform(self.Mz_in_s)
        Wz_h_s = self.pos_variance_transform(self.Mz_h_s)
        bz_s = self.pos_variance_transform(self.pz_s)
        Wr_in_s = self.pos_variance_transform(self.Mr_in_s)
        Wr_h_s = self.pos_variance_transform(self.Mr_h_s)
        br_s = self.pos_variance_transform(self.pr_s)
        Ws_in_s = self.pos_variance_transform(self.Ms_in_s)
        Ws_h_s = self.pos_variance_transform(self.Ms_h_s)
        bs_s = self.pos_variance_transform(self.ps_s)

        z_om, z_os = self.linear_layer(
            (input_m, input_s),
            (h_m, h_s),
            (self.Wz_in_m, Wz_in_s),
            (self.Wz_h_m, Wz_h_s),
            (self.bz_m, bz_s))

        r_om, r_os = self.linear_layer(
            (input_m, input_s),
            (h_m, h_s),
            (self.Wr_in_m, Wr_in_s),
            (self.Wr_h_m, Wr_h_s),
            (self.br_m, br_s))

        # calculate activations
        z_m, z_s = self.sigmoid((z_om, z_os))
        r_m, r_s = self.sigmoid((r_om, r_os))
        # do reset transform
        f_m, f_s = self.elemwise_prod(r_m, r_s, h_m, h_s)

        s_om, s_os = self.linear_layer(
            (input_m, input_s),
            (h_m, h_s),
            (self.Ws_in_m, Ws_in_s),
            (self.Ws_h_m, Ws_h_s),
            (self.bs_m, bs_s))

        s_m, s_s = self.tanh((s_om, s_os))

        # because of independence assertions, this is fine
        # NOTE: 1 - z_s ~ N(1 - z_m, z_s)

        # decompose this sum into two parts
        h_m_1, h_s_1 = self.elemwise_prod(z_m, z_s, s_m, s_s)
        h_m_2, h_s_2 = self.elemwise_prod(1 - z_m, z_s, h_m, h_s)
        h_m_, h_s_ = self.elemwise_sum(h_m_1, h_s_1, h_m_2, h_s_2)
        return h_m_, h_s_

    def elemwise_prod(self, o1_m, o1_s, o2_m, o2_s):
        o_m = (o1_s * o2_m + o2_s * o1_m )/(o1_m + o1_s)
        o_s = (o1_s * o2_s)/(o1_s + o2_s)
        return o_m, o_s

    def elemwise_sum(self, o1_m, o1_s, o2_m, o2_s):
        return o1_m + o2_m, o1_s + o2_s


class GaussianNPRNCell(nn.Module):
    """
        Wrapper around GaussianNPRN that goes through a sequence
    """
    def __init__(self, input_features, hidden_sz, eps=0.0, variant='gru'):
        super(GaussianNPRNCell, self).__init__()
        self.rnn = GaussianNPRN(input_features, hidden_sz, variant='gru')
        self.eps = eps # small value for h_s

    def init_hidden(self, batch_sz):
        h_m = torch.zeros(batch_sz, self.rnn.input_features)
        # non-zero variance?
        h_s = torch.zeros(batch_sz, self.rnn.input_features).fill_(self.eps)
        if has_cuda:
            h_m = h_m.cuda()
            h_s = h_s.cuda()
        return h_m, h_s

    def forward(self, input_seq, hidden=None):
        # input: S x B x 1 x N
        # TODO: implement more checks to make nicer code :)
        if len(input_seq.size()) < 3:
            raise ValueError('input should be of shape S x B x N')
        if hidden is None:
            h_m = autograd.Variable(tt.FloatTensor(input_seq.size(1), self.rnn.hidden_sz).zero_())
            h_s = autograd.Variable(tt.FloatTensor(input_seq.size(1), self.rnn.hidden_sz).fill_(self.eps))
        else:
            h_m, h_s = hidden # tuple
        hiddens_m = []
        hiddens_s = []
        for w in input_seq:
            h_m, h_s = self.rnn(w, h_m, h_s)
            hiddens_m.append(h_m.unsqueeze(0))
            hiddens_s.append(h_s.unsqueeze(0))
        return (torch.cat(hiddens_m, dim=0), torch.cat(hiddens_s, dim=0)), (h_m, h_s)

class GaussianNPRNLanguageModel(nn.Module):
    """
        Language Model wrapper
    """
    def __init__(self, vocab_sz, emb_sz, hidden_sz):
        super(GaussianNPRNLanguageModel, self).__init__()

        self.embeds_layer = nn.Embedding(vocab_sz, emb_sz)
        self.nprn = GaussianNPRNCell(emb_sz, hidden_sz)

        # TODO: implement weight tying
        self.decoder_pre = GaussianNPNLinearLayer(hidden_sz, vocab_sz)
        self.decoder_sigm = GaussianNPNNonLinearity('sigmoid')

    def init_hidden(self, batch_sz):
        return self.nprn.init_hidden(batch_sz)

    def forward(self, input, hidden):
        embeds = self.embeds_layer(input)
        nprn_outs, hiddens = self.nprn(embeds, hidden)
        outs = self.decoder_pre(nprn_outs)
        outs = self.decoder_sigm(outs)
        return outs, hiddens

if __name__ == '__main__':
    # do a basic test
    g = GaussianNPRNCell(300, 128, 0.1, 'gru')
    x = autograd.Variable(tt.FloatTensor(np.random.rand(5,10,300)))
    print(g(x)[0][0].shape)

