import utils
import jieba
import numpy as np
import os

id2w = ['<PAD>', '<UNK>'] + [x[0] for x in utils.LoadCSV('data/wordlist.txt') if int(x[1]) > 2]
w2id = {v: k for k, v in enumerate(id2w)}
id2c = ['<PAD>', '<UNK>'] + [x[0] for x in utils.LoadCSV('data/charlist.txt') if int(x[1]) > 2]
c2id = {v: k for k, v in enumerate(id2c)}

def MakeSen(sen, ssr=True):
	qcache = {}
	sen = sen.lower()
	if sen != '' and not sen in qcache:
		qs = jieba.cut(sen)
		qids = [w2id.get(x, 1) for x in qs][-30:]
		cids = [c2id.get(x, 1) for x in sen][-30:]
		qcache[sen] = (qids, cids)
		if len(qcache) > 1000000: qcache = {}
	else: qids, cids = qcache.get(sen, ([], []))
	if not ssr: return (qids, cids)
	qstr = ','.join(str(x) for x in qids)
	cstr = ','.join(str(x) for x in cids)
	return (qstr, cstr)

def ExtendCandidateQuestionIndexing():
	ret = []
	ext_cands = utils.LoadCSV(r'data/qa_candidate_extend.txt')
	index = max([int(x) for x in np.array(utils.LoadCSV(r'data/train_data.txt'))[:, 0]]) + 1
	for x in ext_cands:
		ret.append([index,x[0],x[1]])
		index+=1
	utils.SaveCSV(ret, r'data/qa_candidate_extend.txt')

def LoadCandidateData():
	ret = []
	for file in os.listdir(r'data/candidates'):
		ret+=(utils.LoadCSV(os.path.join(r'data/candidates', file)))
	return {str(ret.index(x)):[x[0], x[1], MakeSen(x[0], ssr=False)[0], MakeSen(x[0], ssr=False)[1]] for x in ret}

def GenerateInvertedTable(corpus):
	common_words = ['叫','、', '了', '用', '不', '吗', '上', '，', '怎样', '和', '人', '在', '说','能','什么','要', '会','有','是','的','为什么']
	common_chars = ['人','能','要', '不','会', '有', '是','的','为', '什', '么']
	word_inv = {}
	char_inv = {}
	for ind in corpus.keys():
		q = corpus[ind][0]
		ind = str(ind)
		for ch in q:
			if ch in common_chars:
				continue
			if ind not in char_inv.get(ch,[]):
				if ch not in char_inv.keys():
					char_inv[ch] = [ind]
				else:
					char_inv[ch].append(ind)
		for word in jieba.cut(q):
			if word in common_words:
				continue
			if ind not in word_inv.get(word,[]):
				if word not in word_inv.keys():
					word_inv[word] = [ind]
				else:
					word_inv[word].append(ind)
	return [word_inv,char_inv]

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
		ret[x] = ret.get(x,0)+1
	return ret

def GenerateCandidateTFVector(corpus):
	word_tf_tab = {}
	char_tf_tab = {}
	for ind in corpus:
		q = corpus[ind][0]
		word_tf_tab[ind] = ToWordVector(q)
		char_tf_tab[ind] = ToCharVector(q)
	return [word_tf_tab,char_tf_tab]
if __name__ == '__main__':
	pass