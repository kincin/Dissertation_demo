import os
import time
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from transformers import BertTokenizer
import pickle

from model import MMTG
from configs import model_cfgs, data_config as mydata_config
from generate import sample_sequence

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['SECRET_KEY'] = 'mmtg-demo-secret-key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

os.makedirs('config', exist_ok=True)
if not os.path.exists('config/model_config.json'):
    print("创建模型配置文件...")
    with open('config/model_config.json', 'w') as f:
        f.write('''
{
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 250,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "vocab_size": 13317
}
''')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class MMTGModel:
    def __init__(self, model_path, vocab_path, token_id2emb_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        self.data_config = mydata_config()
        self.model_cfgs = model_cfgs

        try:
            print(f"正在加载分词器，路径: {vocab_path}")
            self.tokenizer = BertTokenizer(vocab_file=vocab_path)
            print(f"成功加载分词器，词汇表大小: {len(self.tokenizer.vocab)}")
        except Exception as e:
            print(f"加载分词器时出错: {str(e)}")
            raise
        
        try:
            print(f"正在加载token_id2emb字典，路径: {token_id2emb_path}")
            with open(token_id2emb_path, 'rb') as f:
                self.token_id2emb = pickle.load(f)
            print(f"成功加载token_id2emb字典，大小: {len(self.token_id2emb)}")
        except Exception as e:
            print(f"加载token_id2emb字典时出错: {str(e)}")
            print("将使用随机向量作为嵌入")
            self.token_id2emb = None
        
        try:
            print(f"正在加载模型，路径: {model_path}")
            self.model = MMTG(self.model_cfgs, len(self.tokenizer.vocab), False)  
            self.model.to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_key = k[7:]  
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
                    
            print("已处理状态字典，移除'module.'前缀")
            
            self.model.load_state_dict(new_state_dict)
            print(f"成功加载模型")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        self.model.eval()
    
    def _get_image_embedding(self, image_path):

        if self.token_id2emb is not None:
            keys = list(self.token_id2emb.keys())[:100]  
            embedding = np.mean([self.token_id2emb[k] for k in keys], axis=0)
            return embedding
        else:
            print("使用随机向量作为图像嵌入")
            return np.random.randn(self.data_config.wenlan_emb_size)
    
    def _get_text_embedding(self, text):
        if self.token_id2emb is not None:
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            embeddings = []
            for token_id in token_ids:
                if token_id in self.token_id2emb:
                    embeddings.append(self.token_id2emb[token_id])
            
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                print(f"文本'{text}'没有匹配的token嵌入向量，使用随机向量")
                return np.random.randn(self.data_config.wenlan_emb_size)
        else:
            print("使用随机向量作为文本嵌入")
            return np.random.randn(self.data_config.wenlan_emb_size)
    
    def prepare_input(self, topic, image_text_pairs):
        topic_ids, tpw_attention_mask, tpw_type_ids = self._convert_topic(topic)
        
        img_embs = []
        r_embs = []
        
        for img_path, text in image_text_pairs:
            img_emb = self._get_image_embedding(img_path)
            text_emb = self._get_text_embedding(text)
            
            img_embs.append(img_emb)
            r_embs.append(text_emb)
        
        encoded = [self.tokenizer.convert_tokens_to_ids('[#START#]')]
        
        input_data = {
            'topic_ids': np.asarray(topic_ids),
            'tpw_attention_mask': np.asarray(tpw_attention_mask),
            'tpw_type_ids': np.asarray(tpw_type_ids),
            'topic_emb': self._get_text_embedding(topic),  
            'img_embs': np.asarray(img_embs),
            'r_embs': np.asarray(r_embs),
            'targets': np.asarray(encoded)
        }
        
        return input_data
    
    def _convert_topic(self, topic_words):
        """将主题词转换为 token IDs 和注意力掩码"""
        topic_prompt = "主题词：" + topic_words
        topic_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(topic_prompt))
        attention_mask = [1] * len(topic_ids)
        type_ids = [1] * len(topic_ids)
        
        topic_ids = topic_ids[:self.data_config.topic_prompt_length]
        attention_mask = attention_mask[:self.data_config.topic_prompt_length]
        type_ids = type_ids[:self.data_config.topic_prompt_length]
        
        while len(topic_ids) < self.data_config.topic_prompt_length:
            topic_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            type_ids.append(0)
        
        return topic_ids, attention_mask, type_ids
    
    def generate(self, topic, image_text_pairs, 
                 temperature=1.1, top_k=10, top_p=0.7, 
                 repetition_penalty=1.5, num_samples=3):
        """生成歌词/诗歌"""
        try:
            start_input = self.prepare_input(topic, image_text_pairs)
            results = []
            
            for _ in range(num_samples):
                preds = sample_sequence(
                    self.model,
                    start_input,
                    length=self.data_config.max_seq_length,
                    tokenizer=self.tokenizer,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repitition_penalty=repetition_penalty,
                    device=self.device
                )
                
                preds = [self.tokenizer.convert_ids_to_tokens(line) for line in preds]
                
                all_idx_of_eos = [i for i, v in enumerate(preds) if v == '[#EOS#]']
                
                if len(all_idx_of_eos) >= 10 and '[SEP]' not in preds[:all_idx_of_eos[-1]]:
                    eos_idx = all_idx_of_eos[9]
                    preds = preds[:eos_idx+1] + ['[SEP]']
                elif '[SEP]' in preds:
                    sep_idx = preds.index('[SEP]')
                    preds = preds[:sep_idx+1]
                else:
                    preds = preds + ['[SEP]']
                
                result = ''.join(preds).replace('[SEP]', '').replace('[PAD]', '').replace('[#START#]', '').replace('[#EOS#]', '，')
                
                while result and result[-1] == '，':
                    result = result[:-1]
                
                formatted_result = []
                current_line = ""
                for char in result:
                    current_line += char
                    if char == '，' and len(current_line) > 10: 
                        formatted_result.append(current_line)
                        current_line = ""
                if current_line:
                    formatted_result.append(current_line)
                
                results.append("\n".join(formatted_result))
            
            return results
        except Exception as e:
            print(f"生成过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return [f"生成时出错: {str(e)}"] * num_samples

model = None

def init_model():
    global model
    try:
        model_path = './models/ckpt.pth'  
        vocab_path = './vocab/vocab.txt' 
        token_id2emb_path = './vocab/token_id2emb_dict.pkl'  
        
        if not os.path.exists(vocab_path):
            print(f"错误：词表文件不存在: {vocab_path}")
            raise FileNotFoundError(f"词表文件不存在: {vocab_path}")
        
        if not os.path.exists(model_path):
            print(f"错误：模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        model = MMTGModel(model_path, vocab_path, token_id2emb_path)
    except Exception as e:
        print(f"初始化模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html', model_loaded=(model is not None))

@app.route('/upload', methods=['POST'])
def upload_files():
    if model is None:
        return jsonify({'error': '模型未成功加载，请检查服务器日志'}), 500
        
    if 'files[]' not in request.files:
        return jsonify({'error': '没有文件被上传'}), 400

    files = request.files.getlist('files[]')
    texts = request.form.getlist('texts[]')
    topic = request.form.get('topic', '默认主题')
    
    if len(files) != len(texts):
        return jsonify({'error': '图片和文本数量不匹配'}), 400
    
    image_text_pairs = []
    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{int(time.time())}_{i}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_text_pairs.append((filepath, texts[i] if i < len(texts) else ""))
    
    if not image_text_pairs:
        return jsonify({'error': '没有有效的文件被上传'}), 400
    
    try:
        results = model.generate(topic, image_text_pairs, num_samples=3)
        
        display_data = {
            'topic': topic,
            'pairs': [],
            'results': results
        }
        
        for (img_path, text) in image_text_pairs:
            relative_path = img_path.replace(os.path.join(os.getcwd(), ''), '')
            if relative_path.startswith('/'):
                relative_path = relative_path[1:]
            display_data['pairs'].append({
                'image': relative_path,
                'text': text
            })
        
        return jsonify(display_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'生成过程中出错: {str(e)}'}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    init_model() 
    app.run(debug=True, host='0.0.0.0', port=5000)