# Environment
python=3.8.10
```bash
conda create -n py38 python=3.8.10
```
```bash
conda activate py38 
```


PyTorch = 1.10.1+cu111 (1.7,1.8都试过，附近的版本都可以) 参考来源：https://pytorch.org/get-started/previous-versions/

~~conda install pytorch=1.10.1+cu111 -c pytorch~~
```bash
pip install torch==1.10.1+cu111 torchvision==0.10.2+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

```



OpenNMT-py用文件夹内的版本，内部有修改
```bash
cd OpenNMT-py
pip install -e .
```

# Data
三个数据集iTAPEs、Within-project、Cross-project

# Prepare dataset case(e.g. Vocab)
### 以within数据集为例，GSSIT:
within
```bash
onmt_build_vocab -config title_generation_within_GSSIT.yaml -n_sample 50000 -context
```
Cross-project
```bash
onmt_build_vocab -config title_generation_cross_GSSIT.yaml -n_sample 50000 -context
```


### iTAPE:
within 
```bash
onmt_build_vocab -config title_generation_within_iTAPE.yaml -n_sample 50000
```
Cross-project

```bash
onmt_build_vocab -config title_generation_cross_GSSIT.yaml -n_sample 50000
```

# Train case
-context表示用多个encoder，使用train.src train.des，不加context是单个encoder，只用train.src 

GSSIT:
```bash
onmt_train -config title_generation_within_GSSIT.yaml -context
```
```bash
onmt_train -config title_generation_cross_GSSIT.yaml -context
```

iTAPE:
```bash
onmt_train -config title_generation_within_iTAPE.yaml
```
```bash
onmt_train -config title_generation_cross_iTAPE.yaml
```

# Test case
GSSIT:
```bash
onmt_translate -model model/Within-project/run_GSSIT_step_2000.pt -src data/Within-project/test.src -des data/Within-project/test.des -exp data/Within-project/test.exp -rep data/Within-project/test.rep -oth data/Within-project/test.oth -output data/Within-project/test.pred.GSSIT -gpu 0 -verbose -context
```
```bash
onmt_translate -model model/Within-project/run_GSSIT_step_2000.pt -src data/Within-project/test.src -output data/Within-project/test.pred.GSSIT -gpu 0 -verbose
```
iTAPE:
```bash
onmt_translate -model model/Within-project/run_iTAPE_step_15000.pt -src data/Within-project/test.src  -output data/Within-project/test.pred.iTAPE -gpu 0 -verbose
```

# Evaluate case
GSSIT:
```bash
python utils/EvaluationMetrics.py main ./data/Within-project/test.tgt ./data/Within-project/test.pred.GSSIT
```

iTAPE:
```bash
python utils/EvaluationMetrics.py main ./data/Within-project/test.tgt ./data/Within-project/test.pred.iTAPE
```

# Using Pre-trained Bert to Clasifier the Sentences
pretrained_bert下载路径：https://drive.google.com/drive/folders/1z4zXexpYU10QNlpcSA_UPfMb2V34zHHO
Fine-tune the bert 参考：/bert/run_gssit.sh
Classify the Sentences 参考：/bert/test.sh