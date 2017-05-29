# -*- coding: utf-8 -*-
import jieba
import os
import re
import time
import string

rootpath = "F:\mission-test\\fenlei"
os.chdir(rootpath)  #os.chdir() 方法用于改变当前工作目录到指定的路径
# stopword
words_list = []
filename_list = []
category_list = []
all_words = {}  # 全词库 {'key':value }
stopwords = {}.fromkeys([line.rstrip() for line in open('F:\mission\stopwords.txt')])
category = os.listdir(rootpath)  # 类别列表
delEStr = string.punctuation + ' ' + string.digits #
identify = str.maketrans('', '')


#########################
#       分词，创建词库  #
#########################
def fileWordProcess(contents):
    wordsList = []
    contents = re.sub(r'\s+', ' ', contents)  # trans 多空格 to 空格
    contents = re.sub(r'\n', ' ', contents)  # trans 换行 to 空格
    contents = re.sub(r'\t', ' ', contents)  # trans Tab to 空格
    contents = contents.translate(delEStr)
    for seg in jieba.cut(contents):
        seg = seg.encode('utf8')

        if seg not in stopwords:  # remove 停用词
            if seg != ' ':  # remove 空格
                wordsList.append(seg)  # create 文件词列表
    print(wordsList)
    file_string = ' '.join(str(s) for s in wordsList)
    return file_string


for categoryName in category:  # 循环类别文件，OSX系统默认第一个是系统文件
    if (categoryName == '.DS_Store'): continue
    categoryPath = os.path.join(rootpath, categoryName)  # 这个类别的路径
    filesList = os.listdir(categoryPath)  # 这个类别内所有文件列表
    # 循环对每个文件分词
    for filename in filesList:
        if (filename == '.DS_Store'): continue
        starttime = time.clock()
        contents = open(os.path.join(categoryPath, filename),encoding="UTF-8").read()
        wordProcessed = fileWordProcess(contents)  # 内容分词成列表
        # 暂时不做#filenameWordProcessed = fileWordProcess(filename) # 文件名分词，单独做特征
        #         words_list.append((wordProcessed,categoryName,filename)) # 训练集格式：[(当前文件内词列表，类别，文件名)]
        words_list.append(wordProcessed)
        filename_list.append(filename)
        category_list.append(categoryName)
        endtime = time.clock();
        print('类别:%s >>>>文件:%s >>>>导入用时: %.3f' % (categoryName, filename, endtime - starttime))

# 创建词向量矩阵，创建tfidf值矩阵
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
freWord = CountVectorizer(stop_words='english')
transformer = TfidfTransformer()
fre_matrix = freWord.fit_transform(words_list)#转词向量

tfidf = transformer.fit_transform(fre_matrix)#计算tfidf
import pandas as pd
feature_names = freWord.get_feature_names()  # 特征名
freWordVector_df = pd.DataFrame(fre_matrix.toarray())  # 全词库 词频 向量矩阵
tfidf_df = pd.DataFrame(tfidf.toarray())  # tfidf值矩阵
print(freWordVector_df)
print("cxvxc")
print(tfidf_df)
# print freWordVector_df
tfidf_df.shape


# tf-idf 筛选
tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=False)[:10000].index
print("1111111")
print(len(tfidf_sx_featuresindex))
print("222222")
freWord_tfsx_df = freWordVector_df.ix[:, tfidf_sx_featuresindex]  # tfidf法筛选后的词向量矩阵

df_columns = pd.Series(feature_names)[tfidf_sx_featuresindex]

print(df_columns.shape)
def guiyi(x):
    x[x > 1] = 1
    return x
import numpy as np
tfidf_df_1 = freWord_tfsx_df.apply(guiyi)
tfidf_df_1.columns = df_columns

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
tfidf_df_1['label'] = le.fit_transform(category_list)
tfidf_df_1.index = filename_list

# 卡方检验
from sklearn.feature_selection import SelectKBest, chi2
# ch2 = SelectKBest(chi2, k=int(0.9*len(tfidf_sx_featuresindex)))
ch2 = SelectKBest(chi2, k=700)
nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]
ch2_sx_np = ch2.fit_transform(tfidf_df_1[nolabel_feature],tfidf_df_1['label'])
label_np = np.array(tfidf_df_1['label'])

# 朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

# nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]
# x_train, x_test, y_train, y_test = train_test_split(ch2_sx_np, tfidf_df_1['label'], test_size = 0.2)

X = ch2_sx_np
y = label_np
skf = StratifiedKFold(y, n_folds=10)
y_pre = y.copy()
print(len(skf))
for train_index, test_index in skf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = MultinomialNB().fit(X_train, y_train)
    y_pre[test_index] = clf.predict(X_test)

print('准确率为 %.6f' % (np.mean(y_pre == y)))

# 精准率 召回率 F1score
from sklearn.metrics import confusion_matrix, classification_report

print('precision,recall,F1-score如下：》》》》》》》》')
print(classification_report(y, y_pre))

# confusion matrix
import matplotlib.pyplot as plt
# %matplotlib inline


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category[1:]))
    category_english = ['neibu', 'jimi', 'mimi']
    plt.xticks(tick_marks, category_english, rotation=45)
    plt.yticks(tick_marks, category_english)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')


print('混淆矩阵如下：》》》》》》')
cm = confusion_matrix(y, y_pre)
plt.figure()
plot_confusion_matrix(cm)

plt.show()