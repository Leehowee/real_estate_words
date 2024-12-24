#!/usr/bin/env python3
# coding: utf-8
import os
from pyltp import SentenceSplitter
import jieba.posseg as pseg
import jieba
import math,time
import yaml
          
import matplotlib.pyplot as plt

class SoPmi:
    def __init__(self, config):
        self.train_path = config['train_path']
        self.output_path = config['output_path']
        self.basic_path = config['basic_path']
        self.stop_word_path =  config['stop_word_path']
        self.basic_words = set([line.strip().split('\t')[0] for line in open(self.basic_path)])
        self.stop_words = set([line.strip().split('\t')[0] for line in open(self.stop_word_path)])
        self.all_files = self.traverse_files(self.train_path)


    def traverse_files(self, root_dir):
        """
        通过递归方式深度优先遍历根目录下的所有文件，并打印文件路径
        """
        files = []
        for dir in os.listdir(root_dir):
            cur_dir = os.path.join(root_dir, dir)
            for file in os.listdir(cur_dir):
                file_path = os.path.join(cur_dir, file)
                files.append(file_path)
        return files

    '''分词'''
    def seg_corpus(self, file):
        #将种子词加入到用户词典当中，保证分词能够将种子情感词切开
        for word in self.basic_words:
            jieba.add_word(word)
        # 获取目录下的所有条目（包括文件和子目录）

        # 遍历所有条目，筛选出文件
        seg_data = list()
        count = 0
        # for file in self.all_files:
        #     
        for line in open(file):
            line = line.strip()
            count += 1
            if line:
                seg_data.append([word.word for word in pseg.cut(line) if word.flag[0] not in ['u','w','x','p','q','m']])
            else:
                continue
        return seg_data

    '''统计搭配次数'''
    def collect_cowords(self, seg_data):
        def check_words(sent):
            if self.basic_words.intersection(set(sent)):
                return True
            else:
                return False
        cowords_list = list()
        window_size = 5
        count = 0
        for sent in seg_data:
            count += 1
            if check_words(sent):
                for index, word in enumerate(sent):
                    if index < window_size:
                        left = sent[:index]
                    else:
                        left = sent[index - window_size: index]
                    if index + window_size > len(sent):
                        right = sent[index + 1:]
                    else:
                        right = sent[index: index + window_size + 1]
                    context = left + right + [word]
                    if check_words(context):
                        for index_pre in range(0, len(context)):
                            if check_words([context[index_pre]]):
                                for index_post in range(index_pre + 1, len(context)):
                                    cowords_list.append(context[index_pre] + '@' + context[index_post])
        return cowords_list

    '''计算So-Pmi值'''
    def collect_candiwords(self, seg_data, cowords_list):
        '''互信息计算公式'''
        def compute_mi(p1, p2, p12):
            return math.log2(p12) - math.log2(p1) - math.log2(p2)
        '''统计词频'''
        def collect_worddict(seg_data):
            word_dict = dict()
            all = 0
            for line in seg_data:
                for word in line:
                    if word not in word_dict:
                        word_dict[word] = 1
                    else:
                        word_dict[word] += 1
            all = sum(word_dict.values())
            return word_dict, all
        '''统计词共现次数'''
        def collect_cowordsdict(cowords_list):
            co_dict = dict()
            candi_words = list()
            for co_words in cowords_list:
                candi_words.extend(co_words.split('@'))
                if co_words not in co_dict:
                    co_dict[co_words] = 1
                else:
                    co_dict[co_words] += 1
            return co_dict, candi_words

        '''计算sopmi值'''
        def compute_sopmi(candi_words, basic_words, word_dict, co_dict, all):
            pmi_dict = dict()
            for candi_word in set(candi_words):
                candi_pmi = 0.0
                for basic_word in basic_words:
                    p1 = word_dict[basic_word] / all
                    p2 = word_dict[candi_word] / all
                    pair = basic_word + '@' + candi_word
                    if pair not in co_dict:
                        continue
                    p12 = co_dict[pair] / all
                    candi_pmi += compute_mi(p1, p2, p12)
                    pmi_dict[candi_word] = candi_pmi
            return pmi_dict

        word_dict, all = collect_worddict(seg_data)
        # self.plot_word_cloud(word_dict)
        co_dict, candi_words = collect_cowordsdict(cowords_list)
        words = set(self.basic_words.intersection(set(word_dict.keys())))
        pmi_dict = compute_sopmi(candi_words, words, word_dict, co_dict, all)
        return pmi_dict

    '''保存结果'''
    def save_candiwords(self, pmi_dict, output_path, file):
        fi = open(os.path.join(output_path, str(file)+".txt"), 'w+')

        for word, pmi in sorted(pmi_dict.items(), key=lambda asd:asd[1], reverse=True):
            # 去掉通用词
            if len(word) >= 2 and word not in self.stop_words:
                fi.write(word + ',' + str(pmi) + ',' + str(len(word)) + '\n')
        fi.close()
        return

    def sopmi(self):
        begin_time  = time.time()
        index = 0
        for file in self.all_files:
            start_time  = time.time()
            print('begin process file:%s' % file)
            print('step 1/4:...seg corpus ...')
            seg_data = self.seg_corpus(file)
            end_time1 = time.time()
            print('step 1/4 finished:...cost {0}...'.format((end_time1 - start_time)))
            print('step 2/4:...collect cowords ...')
            cowords_list = self.collect_cowords(seg_data)
            end_time2 = time.time()
            print('step 2/4 finished:...cost {0}...'.format((end_time2 - end_time1)))
            print('step 3/4:...compute sopmi ...')
            pmi_dict = self.collect_candiwords(seg_data, cowords_list)
            end_time3 = time.time()
            print('step 1/4 finished:...cost {0}...'.format((end_time3 - end_time2)))
            print('step 4/4:...save candiwords ...')
            self.save_candiwords(pmi_dict, self.output_path, index)
            end_time = time.time()
            print('finished! cost {0}'.format(end_time - start_time))
            index += 1
        end_time = time.time()
        print('total time cost {0}'.format(end_time - begin_time))

def train(config):
    sopmier = SoPmi(config)
    sopmier.sopmi()

if __name__ == "__main__":
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    train(cfg)
