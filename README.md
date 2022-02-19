# COVID-19 Cough Classification

## Overview

The aim of this project is to classify audio recordings of coughs into COVID-19 positive and negative.

## Installation

### Install required libraries

```
pip install -r requirements.txt
```

## Running the code

```
python app.py
```

## Datasets

### Kaggle Cough-Classifier Dataset

https://www.kaggle.com/himanshu007121/coughclassifier-trial

The major problem with this dataset is, that it is highly unbalanced. Only 19 of the 170 examples are labeled as positive. <br/>
Therefore, in addition to the good test accuracy, the model shows a relatively high false-negative rate. <br/>
More data, especially positive examples, is needed to develop a reliable, cough audio-based Covid-19 test. <br/>

### Virufy Dataset

https://github.com/virufy/virufy_data/tree/main/clinical/segmented

This dataset is provided by the developers of the Virufy app. They offer a webservice (https://virufy.org/en/) which can detect a COVID-19 signature in recordings using an AI algorithm. The open dataset on github contains 121 examples of which 48 are labeled as positive.

### Balanced Dataset

The balanced dataset is created by combining the Kaggle and the Virufy dataset. Since they both contain more negative than positive examples, downsampling is used to create a perfectly balanced dataset.

### Other Datasets

Other datasets have not been investigated, yet.

- **Coswara**: https://github.com/iiscleap/Coswara-Data
- **COUGHVID**: https://zenodo.org/record/4048312

