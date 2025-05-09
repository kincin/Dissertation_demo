from torch.utils.data import Dataset
import numpy as np
import pickle

class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, data_config, if_train=True):
        super(MyDataset, self).__init__()
        self._filename = file_path
        self.data = self.load_data(self._filename)
        self._tokenizer = tokenizer
        self._max_topic_length = data_config.topic_prompt_length
        self._max_sent_length = data_config.max_sent_length
        self._total_len = len(self.data)
        self.if_train = if_train
    
    def load_data(self, data_file):
        f = open(data_file, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        '''
        item.keys:
            'topic', 'topic_emb', 'lyrics', 'rating',
            'text_0', 'text_0_emb', 'text_1', 'text_1_emb', 'text_2', 'text_2_emb', 'text_3', 'text_3_emb', 'text_4', 'text_4_emb',
            'img_0', 'img_0_emb', 'img_1', 'img_1_emb', 'img_2', 'img_2_emb', 'img_3', 'img_3_emb', 'img_4', 'img_4_emb',
            'r_0', 'r_0_emb', 'r_1', 'r_1_emb', 'r_2', 'r_2_emb', 'r_3', 'r_3_emb', 'r_4', 'r_4_emb'
        '''
        topic_emb = self.data[idx]['topic_emb']
        img_embs = [self.data[idx]['img_' + str(i) + '_emb'] for i in range(5)]
        r_embs = [self.data[idx]['r_' + str(i) + '_emb'] for i in range(5)]
        topic_ids, tpw_attention_mask, tpw_type_ids = self.convert_topic(self.data[idx]['topic'])
        targets, attention_mask, type_ids = self.convert_lyrics2ids(self.data[idx]['lyrics']) # a list of list: [[sent1], [sent2], ...]
        batch = {
            'topic_ids': np.asarray(topic_ids),
            'tpw_attention_mask': np.asarray(tpw_attention_mask),
            'tpw_type_ids': np.asarray(tpw_type_ids),
            'topic_emb': np.asarray(topic_emb),
            'img_embs': np.asarray(img_embs),
            'r_embs': np.asarray(r_embs),
            'targets': np.asarray(targets),
            'attention_mask': np.asarray(attention_mask),
            'type_ids': np.asarray(type_ids)
        }
        if self.if_train:
            batch['rating'] = self.data[idx]['rating']
        return batch

    def convert_topic(self, topic_words):
        '''
        topic_words: str of topic words
        '''
        topic_prompt = "主题词：" + topic_words # "Topic words: " + topic_words
        topic_ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(topic_prompt))
        attention_mask = [1] * len(topic_ids)
        type_ids = [1] * len(topic_ids) # the same as the type_ids of the 1st and 5th sentences
        topic_ids = topic_ids[:self._max_topic_length]
        attention_mask = attention_mask[:self._max_topic_length]
        type_ids = type_ids[:self._max_topic_length]
        while len(topic_ids) < self._max_topic_length:
            topic_ids.append(self._tokenizer.pad_token_id)
            attention_mask.append(0)
            type_ids.append(0)
        
        return topic_ids, attention_mask, type_ids


    def convert_lyrics2ids(self, lyrics):
        '''
        lyrics: list of str
        '''
        all_tokens = []
        attention_mask = []
        type_ids = []
        # [[#START#]sent1[#EOS#][#START#]sent2[#EOS#][#START#]sent3[#EOS#][#START#]sent4[#EOS#]]
        for i in range(0, len(lyrics), 2): # i: 0, 2, 4, 6, ...
            for sent in lyrics[i:i+2]:
                tokens = ['[#START#]']
                attention_mask.append(1)
                type_ids.append(0)
                sent = sent.replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', '')
                sent = sent.replace('\xa0', '').replace('\u3000', '')
                sent = self._tokenizer.tokenize(sent)[:self._max_sent_length]
                tokens.extend(sent)
                attention_mask += [1] * len(sent)
                if i == 8: # In order to call back, the type_ids of the 1st and 5th sentences are the same.
                    type_ids += [1] * len(sent)
                else:
                    type_ids += [i//2+1] * len(sent)
                while len(tokens) < self._max_sent_length + 1: # +1 for [#START#]
                    tokens.append(self._tokenizer.pad_token)
                    attention_mask.append(0)
                    type_ids.append(0)
                tokens.append('[#EOS#]')
                attention_mask.append(1)
                type_ids += [0]
                all_tokens += tokens

        all_tokens.append(self._tokenizer.sep_token) # len of all_tokens + [SEP]: max_seq_length + 1
        attention_mask.append(1)
        type_ids += [0]

        all_token_ids = self._tokenizer.convert_tokens_to_ids(all_tokens)

        return all_token_ids, attention_mask, type_ids
