# Identification-and-Severity-Assessment-of-COVID-19-using-Lung-CT-Scans

### Abstract: 
The COVID-19 pandemic, caused by the SARS-CoV-2 virus, continues to have a significant impact on the global population. To effectively triage patients and understand the progression of the disease, a metric-based analysis of diagnostic techniques is necessary. The objective of the present study is to identify COVID-19 from chest CT scans and determine the extent of severity, defined by a severity score that indicates the volume of infection. An unsupervised preprocessing pipeline is proposed to extract relevant clinical features and utilize this information to employ a pretrained ImageNet EfficientNetB5
model to extract discriminative features. Subsequently, a shallow feed-forward neural network is trained to classify the CT scans into three classes, namely COVID-19, Community-Acquired Pneumonia, and Normal. Through various ablation studies, we find that a domain-specific preprocessing pipeline has a significant positive impact on classification accuracy. The infection segmentation mask generated from the preprocessed pipeline performs better than state-of-the-art supervised semantic segmentation models. Further, the estimated infection severity score is observed to be well correlated with radiologists’ assessments. The results confirm the importance of domain-specific pre-processing for training machine learning algorithms.

## How to run the repository

1. Install the neccessary packages required to run the code. \\
	pip install -r requirements.txt