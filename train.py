"""Optional fixed-duration reconstruction; never overwrite included artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import joblib
import numpy as np
import openpyxl
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,get_linear_schedule_with_warmup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from model import ROOT,digest

class Encoded(Dataset):
    def __init__(self,tokens,labels):self.tokens,self.labels=tokens,labels
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):return {**{k:v[i] for k,v in self.tokens.items()},'labels':self.labels[i]}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'retrained_artifacts')
    args=parser.parse_args();directory=args.output.resolve()
    if directory.is_relative_to((ROOT/'artifacts').resolve()):raise ValueError('Cannot overwrite delivered artifacts')
    if directory.exists() and any(directory.iterdir()):raise ValueError('Training output must be empty')
    if not torch.cuda.is_available():raise RuntimeError('Optional BERT training requires a configured CUDA/ROCm GPU')
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'));trainpath=ROOT/'data/train.xlsx'
    if digest(trainpath)!=cfg['training_data_sha256']:raise ValueError('Training data changed')
    workbook=openpyxl.load_workbook(trainpath,read_only=True,data_only=True);rows=workbook['train'].iter_rows(values_only=True)
    if tuple(next(rows))!=('resp_text','clarity'):raise ValueError('Unexpected training header')
    data=list(rows);workbook.close();texts=[str(r[0]) for r in data];labels=[r[1] for r in data]
    classes=cfg['class_order'];ids={label:i for i,label in enumerate(classes)}
    if len(data)!=cfg['training_rows']:raise ValueError('Training row count changed')
    seed=cfg['bert_training']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    revision=cfg['pretrained_revision'];base=cfg['pretrained_model_id']
    tokenizer=AutoTokenizer.from_pretrained(base,revision=revision)
    model=AutoModelForSequenceClassification.from_pretrained(base,revision=revision,num_labels=3,label2id=ids,id2label={i:x for x,i in ids.items()}).to('cuda:0')
    tokens=tokenizer(texts,truncation=True,max_length=512,padding=False,add_special_tokens=True)
    ds=Encoded(tokens,[ids[x] for x in labels]);collator=DataCollatorWithPadding(tokenizer,return_tensors='pt')
    options=cfg['bert_training'];per_epoch=(len(ds)//2)//4
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=round(per_epoch*2*options['warmup_ratio']),num_training_steps=per_epoch*2)
    step=0;epoch=1;model.train()
    while step<options['optimizer_steps']:
        order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(seed+epoch*100003)).tolist()
        batches=[order[i:i+2] for i in range(0,len(order),2)]
        loader=DataLoader(ds,batch_sampler=batches,num_workers=0,collate_fn=collator,generator=torch.Generator().manual_seed(seed))
        optimizer.zero_grad(set_to_none=True);accumulated=0
        for batch in loader:
            batch={k:v.to('cuda:0') for k,v in batch.items()};(model(**batch).loss/4).backward();accumulated+=1
            if accumulated==4:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);accumulated=0;step+=1
                if step%128==0:print(f'OPTIONAL TRAINING {step}/{options["optimizer_steps"]}',flush=True)
                if step>=options['optimizer_steps'] or step%per_epoch==0:break
        epoch+=1
    directory.mkdir(parents=True,exist_ok=True);model.save_pretrained(directory/'bert',safe_serialization=True);tokenizer.save_pretrained(directory/'bert')
    vcfg=dict(cfg['tfidf']['vectorizer']);vcfg['ngram_range']=tuple(vcfg['ngram_range'])
    linear=Pipeline([('vectorizer',TfidfVectorizer(**vcfg)),('classifier',LogisticRegression(**cfg['tfidf']['classifier']))])
    with threadpool_limits(limits=4):linear.fit(texts,labels)
    joblib.dump(linear,directory/'tfidf.joblib')
    pipeline=json.loads((ROOT/'artifacts/pipeline.json').read_text(encoding='utf-8'));(directory/'pipeline.json').write_text(json.dumps(pipeline,indent=2)+'\n',encoding='utf-8')
    hashes={str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file()}
    (directory/'manifest.json').write_text(json.dumps(dict(status='retrained',files_sha256=hashes,training_rows=len(data),synthetic_rows=0,bitwise_reproduction_guaranteed=False),indent=2)+'\n',encoding='utf-8')
    print(f'Optional retrained ensemble saved to {directory}; delivered artifacts preserved',flush=True)

if __name__=='__main__':main()
