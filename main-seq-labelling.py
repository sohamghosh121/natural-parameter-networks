from torch.autograd import Variable
from torch import nn
import torch.optim as O
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from nprn import GaussianNPRNClassifier

def to_onehot(sz, tensor):
    bsz = tensor.size(0)
    target_onehot = torch.zeros(bsz, sz)
    if tensor.is_cuda:
        target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, tensor, 1)
    return Variable(target_onehot)

# Define the fields associated with the sequences.
WORD = data.Field(init_token="<bos>", eos_token="<eos>")
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# We can also define more than two columns.
WORD = data.Field(init_token="<bos>", eos_token="<eos>")
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")
PTB_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# Load the specified data.
train, val, test = datasets.UDPOS.splits(
    fields=(('word', WORD), ('udtag', UD_TAG), ('ptbtag', PTB_TAG)),
    path="./data/en-ud-v2")

WORD.build_vocab(train.word, val.word, test.word, vectors=[GloVe(name='6B', dim='300')])
UD_TAG.build_vocab(train.udtag)
PTB_TAG.build_vocab(train.ptbtag)

train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=32)

ptbmodel = GaussianNPRNClassifier(len(WORD.vocab),
    300,
    len(PTB_TAG.vocab),
    seq_labeling=True,
    pretrained_emb=WORD.vocab.vectors)

udmodel = GaussianNPRNClassifier(len(WORD.vocab),
    300,
    len(UD_TAG.vocab),
    seq_labeling=True,
    pretrained_emb=WORD.vocab.vectors)

if torch.cuda.is_available():
    ptbmodel = ptbmodel.cuda()
    udmodel = udmodel.cuda()

ptbmodel_optim = O.Adagrad(ptbmodel.parameters())
udmodel_optim = O.Adagrad(udmodel.parameters())


criterion = nn.BCELoss(size_average=False)

print("There are %d PTB tags" % len(PTB_TAG.vocab))
print("There are %d UD tags" % len(UD_TAG.vocab))

log_interval = 10

def run_model(batch_iter, epoch, train=True):
    ptb_preds = open('seq-labeling-results/ptb-preds-%d.txt' % epoch, 'w')
    ud_preds = open('seq-labeling-results/ud-preds-%d.txt' % epoch, 'w')
    ptb_preds_uncertainties = open('seq-labeling-results/ptb-uncertainties-%d.txt' % epoch, 'w')
    ud_preds_uncertainties = open('seq-labeling-results/ud-uncertainties-%d.txt' % epoch, 'w')

    if train:
        print("""~~~~~~~~~~~~  TRAIN ~~~~~~~~~~~~""")
    else:
        print("""~~~~~~~~~~~~  VALID  ~~~~~~~~~~~~""")
    tot_ptbloss = 0
    tot_udloss = 0
    tot_ptbcorrect = 0
    tot_udcorrect = 0
    tot_ex_ptb = 0
    tot_ex_ud = 0

    tot_batch = 0
    # train

    batch_ix = 1
    for batch in iter(batch_iter):
        ptbtag = batch.ptbtag
        udtag = batch.udtag
        word = batch.word

        ptbtarget = ptbtag.view(-1, 1) # flatten sequence
        udtarget = udtag.view(-1, 1) # flatten sequence
        ptbtarget_1hot = to_onehot(len(PTB_TAG.vocab), ptbtarget)
        udtarget_1hot = to_onehot(len(UD_TAG.vocab), udtarget)
        data = Variable(word)

        # train ptbmodel
        hidden = ptbmodel.init_hidden(batch.batch_size)
        ptbmodel_optim.zero_grad()
        ptb_outputs_m, ptb_outputs_s = ptbmodel(data, hidden)
        ptb_outputs_flat = ptb_outputs_m.view(-1, len(PTB_TAG.vocab))
        ptb_loss = criterion(ptb_outputs_flat, ptbtarget_1hot)
        tot_ptbloss += ptb_loss.data.item()
        _, ptb_pred = ptb_outputs_flat.max(1, keepdim=True)
        is_correct = ptbtarget.eq(ptb_pred)
        tot_ptbcorrect += is_correct.sum().item()
        if train:
            ptb_loss.backward()
            ptbmodel_optim.step()

        # train udmodel
        hidden = udmodel.init_hidden(batch.batch_size)
        udmodel_optim.zero_grad()
        ud_outputs_m, ud_outputs_s = udmodel(data, hidden)
        ud_outputs_flat = ud_outputs_m.view(-1, len(UD_TAG.vocab))
        ud_loss = criterion(ud_outputs_flat, udtarget_1hot  )
        _, ud_pred = ud_outputs_flat.max(1, keepdim=True)
        is_correct = udtarget.eq(ud_pred)
        tot_udcorrect += is_correct.sum().item()
        tot_ptbloss += ptb_loss.data.item()
        tot_udloss += ud_loss.data.item()
        if train:
            ud_loss.backward()  
            udmodel_optim.step()

        tot_ex_ptb += ptb_outputs_flat.size(0)
        tot_ex_ud += ud_outputs_flat.size(0)

        tot_batch += batch.batch_size
        if not train:
            pred = torch.argmax(ptb_outputs_m, 2)
            for s_ix in range(ptb_outputs_m.size(1)):
                for w_ix in range(ptb_outputs_m.size(0)):
                    pr = pred[w_ix, s_ix].item()
                    gt = ptbtag[w_ix, s_ix].item()
                    ptb_preds.write(PTB_TAG.vocab.itos[pr] + ' ' + PTB_TAG.vocab.itos[gt] + '\n')
                    ptb_preds_uncertainties.write('%d %d %.4f\n' % (pr, gt, ptb_outputs_s[w_ix, s_ix, pr]))
                ptb_preds.write('\n')

            pred = torch.argmax(ud_outputs_m, 2)
            for s_ix in range(ud_outputs_m.size(1)):
                for w_ix in range(ud_outputs_m.size(0)):
                    pr = pred[w_ix, s_ix].item()
                    gt = udtag[w_ix, s_ix].item()
                    ud_preds.write(UD_TAG.vocab.itos[pr] + ' ' + UD_TAG.vocab.itos[gt] + '\n')
                    ud_preds_uncertainties.write('%d %d %.4f\n' % (pr, gt, ud_outputs_s[w_ix, s_ix, pr]))
                ud_preds.write('\n')
        if batch_ix % 100 == 0:
            print('[PTB] Loss: %.3f    Accuracy: %.3f' % (tot_ptbloss / tot_batch, float(tot_ptbcorrect) / float(tot_ex_ptb)))
            print('[ UD] Loss: %.3f    Accuracy: %.3f' % (tot_udloss / tot_batch, float(tot_udcorrect) / float(tot_ex_ud)))

        batch_ix += 1

    print('-' * 89)
    print('[PTB] Loss: %.3f    Accuracy: %.3f' % (tot_ptbloss / tot_batch, float(tot_ptbcorrect) / float(tot_ex_ptb)))
    print('[ UD] Loss: %.3f    Accuracy: %.3f' % (tot_udloss / tot_batch, float(tot_udcorrect) / float(tot_ex_ud)))
    print('-' * 89)

val_freq = 10
run_model(val_iter, 0, train=False)
for epoch in range(1, 1000):
    print('##### Epoch %d' % epoch)
    run_model(train_iter, epoch, train=True)
    if epoch % val_freq == 0:
        run_model(val_iter, epoch, train=False)