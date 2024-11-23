# 🍊 FinCSE-Multilingual
- **Multilingual Financial SimCSE** for searching financial/economical sentences.
- Fine-tuned on financial sentence pairs in six languages (KO, EN, ZN, JA, VI, ID), while remaining functional for other pre-trained languages.
- Reference : [SimCSE Paper](https://aclanthology.org/2021.emnlp-main.552/), [Original Code](https://github.com/princeton-nlp/SimCSE)


## 1. Model
- The FinCSEs are built upon multilingual PLMs, such as **mBERT** and **XLM-RoBERTa**.
- The FinCSE code in this repository is not derived from the original SimCSE code but is newly implemented.

<img src="simcse.PNG" alt="example image" width="600" height="200"/>

## 2. Training Data
- The FinCSEs are fine-tuned using the '[Financial Domain Multilingual Parallel Corpus (금융 분야 다국어 병렬 말뭉치 데이터)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71782)' provided by AI HUB in South Korea.
- In this corpus, approximately **2.5 million English, Chinese, Japanese, Vietnamese**, and **Indonesian** sentences in the financial domain are paired with their corresponding **Korean** sentences.
```
(KO) 이처럼 금융상품의 경우 판매단계에서 금융회사의 ... 상품의 권유는 기본이고 필수라 할 것이다.
(ZN) 像这样，金融商品在销售阶段，提供金融公司适当的信息和推荐适合金融消费者的商品是基本，也是必须的.
```
- Since this corpus doesn't have similarity scores like a general STS dataset, performance evaluation is based on a **sentence retrieval** task, which searches for corresponding sentences in other languages.

&nbsp;&nbsp; (※ Note that this corpus is authorized only for individuals with South Korean citizenship.)


## 3. Implementation
You can fine-tune your FinCSE model using your own parallel sentence data.

**(1) Preparing Dataset**
- The dataset must be placed in the **'data'** directory and organized in the following structure.
- The headers and the following parallel sentences must be delimited with '\t' (tab-separated).
```
sent0  sent1
(...)  (...)
```

**(2) Fine-tuning**
- Fine-tune your FinCSE by executing the shell script **'run_train.sh'** in the **'train'** directory.
- You can modify the base model, dataset path, or hyperparameters in the shell script.
```
cd train
sh run_train.sh
```

**(3) Evaluation**
- Evaluate your FinCSE by executing the shell script **'run_eval.sh'** in the **'evaluation'** directory.
- You can modify the model to be evaluated, the dataset path, or the hyperparameters in the shell script.
```
cd evaluation
sh run_eval.sh
```

**(4) Inference**
- 



