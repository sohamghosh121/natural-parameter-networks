from torch.autograd import Variable
from torch import nn
import torch.optim as O
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from nprn import GaussianNPRNClassifier

torch.manual_seed(42)

criterion = nn.BCELoss(size_average=False)

def to_onehot(sz, tensor):
    bsz = tensor.size(0)
    target_onehot = torch.zeros(bsz, sz)
    if tensor.is_cuda:
        target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, tensor, 1)
    return Variable(target_onehot)


# Approach 1:
# set up fields
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)


# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)

# build the vocabulary
TEXT.build_vocab(train.text, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train.label)

# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_size=8, device="cuda:0", sort=False)

imdb_model = GaussianNPRNClassifier(len(TEXT.vocab),
    300,
    len(LABEL.vocab),
    seq_labeling=False,
    pretrained_emb=TEXT.vocab.vectors)

if torch.cuda.is_available():
    imdb_model = imdb_model.cuda()

imdb_model_optim = O.Adam(imdb_model.parameters(), lr=1e-3, weight_decay=1e-4)

def run_model(batch_iter, epoch, train=True):
    preds_out = open('imdb-sentiment/imdb-preds-%d.txt' % epoch, 'w')

    if train:
        print("""~~~~~~~~~~~~  TRAIN ~~~~~~~~~~~~""")
    else:
        print("""~~~~~~~~~~~~  VALID  ~~~~~~~~~~~~""")
    tot_loss = 0
    tot_ex = 0
    tot_correct = 0
    tot_batch = 0
    # train

    batch_ix = 1
    for batch in iter(batch_iter):
        text = batch.text[0].t()

        if text.size(0) > 1000:
            continue # skip long reviews

        label = batch.label.unsqueeze(1) # unsqueeze to keepdim=True
        label = label

        label_target = to_onehot(len(LABEL.vocab), label)
        data = Variable(text)

        # train imdb_model
        hidden = imdb_model.init_hidden(batch.batch_size)
        imdb_model_optim.zero_grad()
        output_m, output_s = imdb_model(data, hidden)
        loss = criterion(output_m, label_target)
        tot_loss += loss.data.item()
        pred = torch.argmax(output_m, 1, keepdim=True)
        is_correct = label.eq(pred)
        tot_correct += is_correct.sum().item()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm(imdb_model.parameters(), 0.2)
            imdb_model_optim.step()

        tot_ex += output_m.size(0)
        tot_batch += batch.batch_size

        preds = torch.argmax(output_m, 1)

        if not train:
            for s_ix in range(output_m.size(1)):
                pr = preds[s_ix].item()
                gt = label[s_ix].item()
                preds_out.write('%d %d %.4f\n' % (pr, gt, output_s[s_ix, pr].item()))

        tot_batch += batch.batch_size
        torch.cuda.empty_cache()
        if batch_ix % 50 == 0:
            print('Loss: %.3f    Accuracy: %.3f' % (tot_loss / tot_batch, float(tot_correct) / float(tot_ex)))

        batch_ix += 1

    print('-' * 89)
    print('Loss: %.3f    Accuracy: %.3f' % (tot_loss / tot_batch, float(tot_correct) / float(tot_ex)))
    print('-' * 89)

val_freq = 1
# run_model(test_iter, 0, train=False)
for epoch in range(1, 1000):
    print('##### Epoch %d' % epoch)
    run_model(train_iter, epoch, train=True)
    if epoch % val_freq == 0:
        run_model(test_iter, epoch, train=False)