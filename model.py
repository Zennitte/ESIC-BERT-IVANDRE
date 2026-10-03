"""Self-contained BERT + TF-IDF 50/50 probability ensemble."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

class Ensemble:
    def __init__(self,path=ROOT/'artifacts',device='cpu',verify=True):
        self.path=Path(path)
        if verify:
            manifest=json.loads((self.path/'manifest.json').read_text(encoding='utf-8'))
            for name,sha in manifest['files_sha256'].items():
                if digest(self.path/name)!=sha:raise ValueError(f'Artifact hash differs: {name}')
        self.cfg=json.loads((self.path/'pipeline.json').read_text(encoding='utf-8'))
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable; use --device cpu')
        self.device=torch.device('cuda:0' if device=='cuda' else device)
        self.tokenizer=AutoTokenizer.from_pretrained(self.path/self.cfg['bert'],local_files_only=True)
        self.bert=AutoModelForSequenceClassification.from_pretrained(self.path/self.cfg['bert'],local_files_only=True).to(self.device).eval()
        self.tfidf=joblib.load(self.path/self.cfg['tfidf'])
        if self.cfg['component_order']!=['bert','tfidf'] or self.cfg['weights']!=[0.5,0.5]:raise ValueError('Expected fixed 50/50 dual ensemble')
        if set(self.tfidf.classes_)!=set(self.cfg['class_order']):raise ValueError('TF-IDF classes differ')
        if [self.bert.config.id2label[i] for i in range(3)]!=self.cfg['class_order']:raise ValueError('BERT class order differs')

    def predict_proba(self,texts,batch_size=4):
        if batch_size<1:raise ValueError('batch_size must be positive')
        output=[];order=[list(self.tfidf.classes_).index(label) for label in self.cfg['class_order']]
        for start in range(0,len(texts),batch_size):
            batch=[x if isinstance(x,str) else str(x) for x in texts[start:start+batch_size]]
            tokens=self.tokenizer(batch,truncation=True,max_length=self.cfg['max_length'],padding=True,return_tensors='pt')
            with torch.inference_mode():
                b=torch.softmax(self.bert(**{k:v.to(self.device) for k,v in tokens.items()}).logits,-1).float().cpu().numpy()
            t=self.tfidf.predict_proba(batch)[:,order];p=0.5*b+0.5*t
            if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
            output.append(p)
        return np.concatenate(output) if output else np.empty((0,3))

    def predict(self,texts,batch_size=4):
        return [self.cfg['class_order'][i] for i in self.predict_proba(texts,batch_size).argmax(1)]
