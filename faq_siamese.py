from keras.callbacks import ModelCheckpoint

import faq_preprocess as pp
import utils
import jieba,random
import h5py
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
import math
from keras.utils import plot_model


# Hyper parameter
nwords = len(pp.id2w)
nchars = len(pp.id2c)
batch_size = 64

# Load data

# Candidate qas
# Key: Index, values: question, answer, word vector, char vector
candidate = pp.LoadCandidateData()

# Inverted table for prediction
w_inv_tab, c_inv_tab = pp.GenerateInvertedTable(candidate)

# Candidate questions to tf vectors
word_idf_tab, char_idf_tab = pp.GenerateCandidateTFVector(candidate)


# Siamese model part
def EuclideanDistance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def EuclideanDistanceOutputShape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def ContrastiveLoss(y_true, y_pred):
    # Contrastive loss from http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def Accuracy(y_true, y_pred):
    # Compute classification accuracy with a fixed threshold on distances.
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


uw_input = Input(shape=(30,), dtype='int32')
uc_input = Input(shape=(30,), dtype='int32')

with h5py.File('data/w2v.h5', 'r') as dfile: w2v = dfile['w2v'][:]
w_emb_layer = Embedding(nwords, 50, weights=[w2v], trainable=True)
c_emb_layer = Embedding(nchars, 50)

uw_emb = w_emb_layer(uw_input)
uc_emb = c_emb_layer(uc_input)

# new cnn 1d model
xw = Conv1D(64, 5, strides=2, padding='valid', activation='relu')(uw_emb)
xw = Conv1D(64, 5, strides=2, padding='valid', activation='relu')(xw)
xw = GlobalAveragePooling1D()(xw)
xc = Conv1D(64, 5, strides=2, padding='valid', activation='relu')(uc_emb)
xc = Conv1D(64, 5, strides=2, padding='valid', activation='relu')(xc)
xc = GlobalAveragePooling1D()(xc)
embed = Concatenate()([xw, xc])

sen_model = Model([uw_input, uc_input], embed)
sen_model.summary()
uw_input = Input(shape=(30,), dtype='int32')
uc_input = Input(shape=(30,), dtype='int32')
vw_input = Input(shape=(30,), dtype='int32')
vc_input = Input(shape=(30,), dtype='int32')

uu = sen_model([uw_input, uc_input])
vv = sen_model([vw_input, vc_input])

distance = Lambda(EuclideanDistance,
                  output_shape=EuclideanDistanceOutputShape)([uu, vv])

model = Model([uw_input, uc_input, vw_input, vc_input], distance)
model.compile(loss=ContrastiveLoss, optimizer=Adam(0.001), metrics=[Accuracy])
model.summary()

# plot_model(sen_model, to_file='model.png')
mfile = r'models/faq_siamese.h5'

try:
    model.load_weights(mfile)
except:
    print('new model')


# functions for training

data = utils.LoadCSV('data/train_data.txt')

def NoiseAdd(sen):
    for i in range(2):
        if random.random() > 0.5:
            sen += random.choice(['是谁', '怎么', '怎么样', '吗'])
        if random.random() > 0.5:
            sen = random.choice(['谁是', '怎么', '怎么样', '谁知道']) + sen
    return sen


def Generator():
    while True:
        Xs = [np.zeros((batch_size, 30)) for _ in range(4)]
        Y = np.zeros((batch_size, 1))
        for ii in range(batch_size):
            u, v, y = random.sample(data, 1)[0]
            uu = tuple(pp.MakeSen(u, False))
            vv = tuple(pp.MakeSen(v, False))
            Y[ii] = y
            Xs[0][ii][:len(uu[0])] = uu[0]
            Xs[1][ii][:len(uu[1])] = uu[1]
            Xs[2][ii][:len(vv[0])] = vv[0]
            Xs[3][ii][:len(vv[1])] = vv[1]
        yield Xs, Y

def train():
    saver = ModelCheckpoint(mfile, monitor='val_loss', save_best_only=True, save_weights_only=True)
    gen = Generator()
    val_data = next(gen)
    model.fit_generator(gen, steps_per_epoch=600, epochs=15, validation_data=val_data, callbacks=[saver])


# Functions for prediction
word_idf = utils.LoadDict(r'data/wordlist.txt', float)
char_idf = utils.LoadDict(r'data/charlist.txt', float)


def ToWordVector(Q):
    ws = jieba.cut(Q)
    ret = {}
    for x in ws:
        ret[x] = ret.get(x, 0) + 1
    return ret


def ToCharVector(Q):
    cs = [x for x in Q if utils.IsChsStr(x)]
    ret = {}
    for x in cs:
        ret[x] = ret.get(x, 0) + 1
    return ret


def TF_Sim(v1, v2):
    l1 = math.sqrt(sum([x * x for x in v1.values()]))
    l2 = math.sqrt(sum([x * x for x in v2.values()]))
    ll = l1 * l2 + 1e-10
    r = 0
    for x, y in v1.items():
        r += y * v2.get(x, 0)
    return r / ll


def ModelPred(tdata):
    badends = '?？啊呀呢'
    tdata = tdata.strip(badends)
    uw = ToWordVector(tdata)
    uc = ToCharVector(tdata)
    u = pp.MakeSen(tdata, ssr=False)
    selected_data = []
    selected_inds = []
    cut = jieba.cut(tdata)
    for x in cut:
        for ind in w_inv_tab.get(x, []):
            selected_data.append(candidate[ind])
            selected_inds.append(ind)
    if len(selected_data) == 0:
        for x in tdata:
            for ind in c_inv_tab.get(x, []):
                selected_data.append(candidate[ind])
                selected_inds.append(ind)
    if len(selected_data) == 0:
        return [[0, 'no matching question', 'no matching answer']]
    inum = len(selected_data)
    Xs = [np.zeros((inum, 30)) for _ in range(4)]
    ii = 0
    for v in selected_data:
        Xs[0][ii][:len(u[0])] = u[0]
        Xs[1][ii][:len(u[1])] = u[1]
        Xs[2][ii][:len(v[2])] = v[2]
        Xs[3][ii][:len(v[3])] = v[3]
        ii += 1
    pred = model.predict(Xs, batch_size=1024)
    pred = 1 - pred.reshape((1, -1))[0]
    word_idf_sim = []
    char_idf_sim = []
    for ind in selected_inds:
        qw = word_idf_tab[ind]
        qc = char_idf_tab[ind]
        word_idf_sim.append(TF_Sim(qw, uw))
        char_idf_sim.append(TF_Sim(qc, uc))
    final_pred_ret = []
    for i in range(inum):

        final_pred_ret.append(0.3 * pred[i] + 0.35 * word_idf_sim[i] + 0.35 * char_idf_sim[i])
    lst_final_pred_ret = list(enumerate(final_pred_ret))
    lst_final_pred_ret = sorted(lst_final_pred_ret, key=lambda x: x[1], reverse=True)
    lst_final_pred_ret = [[x[1], selected_data[x[0]][0], selected_data[x[0]][1]] for x in lst_final_pred_ret]
    return lst_final_pred_ret


def ShowCandidate(q):
    pred = ModelPred(q)[:5]
    ret = {}
    for x in pred:
        if x[1] in ret.keys():
            continue
        else:
            ret[x[1]] = x
    return [x for x in ret.values()]


def Answer(q):
    return ModelPred(q)[0]


if __name__ == '__main__':
    # train()
    for x in ShowCandidate('为什么植物会指南'):
        print(x)
