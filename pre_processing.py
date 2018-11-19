#coding:utf-8
import sys
import os
import jieba # pip install jieba

# input files
train_file = './text_classification_data/cnews.train.txt'
val_file = './text_classification_data/cnews.val.txt'
test_file = './text_classification_data/cnews.test.txt'

# output files
seg_train_file = './text_classification_data/cnews.train.seg.txt'
seg_val_file = './text_classification_data/cnews.val.seg.txt'
seg_test_file = './text_classification_data/cnews.test.seg.txt'

vocab_file = './text_classification_data/cnews.vocab.txt'
category_file = './text_classification_data/cnews.category.txt'



# 生成分词后的文件
def generate_seg_file(input_file, output_seg_file):
    """Segment the sentences in each line in input_file"""
    with open(input_file, 'rb') as f:
        lines = f.readlines()
    with open(output_seg_file, 'wb') as f:
        for line in lines:
            label, content = line.decode('utf-8').strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            f.write(out_line.encode('utf-8'))


def generate_vocab_file(input_seg_file, output_vocab_file):
    with open(input_seg_file, 'rb') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        label, content = line.decode('utf-8').strip('\r\n').split('\t')
        for word in content.split():
            word_dict.setdefault(word, 0) # 字典中如果没有word，则添加进去，并将词频设置为0；如果有word，这句就不会起作用
            word_dict[word] += 1
            
    # [(word, frequency), ..., ()]
    sorted_word_dict = sorted(word_dict.items(), key = lambda d:d[1], reverse=True)
    
    with open(output_vocab_file, 'wb') as f:
        f.write('<UNK>\t10000000\n'.encode('utf-8'))
        for item in sorted_word_dict:
            out_line = '%s\t%d\n' % (item[0], item[1])
            f.write(out_line.encode('utf-8'))


def generate_category_dict(input_file, category_file):
    with open(input_file, 'rb') as f:
        lines = f.readlines()
    category_dict = {}
    for line in lines:
        label, content = line.decode('utf-8').strip('\r\n').split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] += 1
    category_number = len(category_dict)
    with open(category_file, 'wb') as f:
        for category in category_dict:
            line = '%s\n' % category
            print ('%s\t%d' % (category, category_dict[category]) )
            f.write(line.encode('utf-8'))
            

def main():
    # with open(val_file, 'rb') as f:
    #     lines = f.readlines()
    # print(lines[0].decode('utf-8'))

    # label, content = lines[0].decode('utf-8').strip('\r\n').split('\t')
    # print (content)

    # word_iter = jieba.cut(content)
    # print ('/ '.join(word_iter))

    generate_seg_file(train_file, seg_train_file)
    generate_seg_file(val_file, seg_val_file)
    generate_seg_file(test_file, seg_test_file)

    generate_vocab_file(seg_train_file, vocab_file)

    generate_category_dict(train_file, category_file)


if __name__ == '__main__':
    main()

