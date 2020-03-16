from torchtext.data import Field
import torchtext
import torch
from torch.autograd import Variable
import jieba
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from flask import Flask, request, jsonify
class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, labels, weight, **kwargs):
        super(textCNN, self).__init__(**kwargs)
        self.labels = labels
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
        self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
        self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
        self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
        self.linear = nn.Linear(3, labels)

    def forward(self, inputs):
        inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
        x1 = F.relu(self.conv1(inputs))
        x2 = F.relu(self.conv2(inputs))
        x3 = F.relu(self.conv3(inputs))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), -1)
        x = x.view(inputs.shape[0], 1, -1)

        x = self.linear(x)
        x = x.view(-1, self.labels)

        return(x)
def read_data_tsv(filename):
    f=open(filename,encoding='utf-8')
    labels=[]
    text=[]

    for line in f.readlines():
        if line.find('text')!=-1:
            continue
        _,label,content=line.split('\t')
        content_seg=jieba.lcut(content)
        labels.append(int(label))
        text.append(content_seg)
    return text,labels

def tokenize(x):
    return [item for item in x.split(' ')]

model=None
TEXT=None
def load():
    train_new_file='./data_new/train.tsv'
    test_new_file='./data_new/test.tsv'
    #print(tokenize(x_text[0]))
    global TEXT
    import os
    if not os.path.exists('.vector_cache'):
        os.mkdir('.vector_cache')
    from torchtext.vocab import Vectors
    vectors=Vectors('sgns.zhihu.word')
    x_train,y_train=read_data_tsv(train_new_file)
    x_test,y_test=read_data_tsv(test_new_file)
    TEXT=Field(sequential=True,fix_length=25)
    TEXT.build_vocab(x_train,x_test,vectors=vectors)
    LABEL=Field(sequential=False, use_vocab=False)
    LABEL.build_vocab(y_train,y_test)
    print("loaded text field")
    global model
    model_torch=torch.load('cnn_epo_9.pkl',map_location='cpu')
    model=model_torch
    print('model loaded')
def get_cut(sentence):
    ls_sentence=jieba.lcut(sentence)
    return [ls_sentence]
def to_result(result):
    result=result.squeeze()
    result_np=result.detach().numpy()
    result_id=np.argmax(result_np)
    return result_id
def test_model():
    str_in = '非常非常实用，既能拉货还能代步，最多拉过2吨货物'
    print(str_in)
    print(get_cut(str_in))
    str_in_preprocess = TEXT.process(get_cut(str_in))
    str_in_preprocess = str_in_preprocess.permute(1, 0)
    result = model(str_in_preprocess)
    print(to_result(result))
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_in=request.form['input']
        print('got text:',text_in)
        input=get_cut(text_in)
        input_preprocess=TEXT.process(input)
        input_preprocess=input_preprocess.permute(1,0)
        result=to_result(model(input_preprocess))
        print('predict result:',result)
        return jsonify({'result':str(result)})
if __name__=="__main__":
    load()
    app.run(host='0.0.0.0')

