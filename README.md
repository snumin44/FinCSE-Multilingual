# 🍊 FinCSE-Multilingual
- **Multilingual Financial SimCSE** for searching financial/economical sentences.
- Fine-tuned on financial sentence pairs in five languages (KO, EN, ZN, JA, VI), while remaining functional for other pre-trained languages.
- Reference : [SimCSE Paper](https://aclanthology.org/2021.emnlp-main.552/), [Original Code](https://github.com/princeton-nlp/SimCSE)


## 1. Model
- The FinCSEs are built upon multilingual PLMs, such as **mBERT** and **XLM-RoBERTa**.
- The FinCSE code in this repository is not derived from the original SimCSE code but is newly implemented.

<img src="simcse.PNG" alt="example image" width="600" height="200"/>

## 2. Training Data
- The FinCSEs are fine-tuned using the '[Financial Domain Multilingual Parallel Corpus (금융 분야 다국어 병렬 말뭉치 데이터)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71782)' provided by AI HUB in South Korea.
- In this corpus, approximately **2.5 million** English, Chinese, Japanese, and Vietnamese sentences in the financial domain are paired with their corresponding Korean sentences.
```

```
