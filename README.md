# Identification-and-Severity-Assessment-of-COVID-19-using-Lung-CT-Scans
---
### Abstract: 
#### The COVID-19 pandemic, caused by the SARS-CoV-2 virus, continues to have a significant impact on the global population. To effectively triage patients and understand the progression of the disease, a metric-based analysis of diagnostic techniques is necessary. The objective of the present study is to identify COVID-19 from chest CT scans and determine the extent of severity, defined by a severity score that indicates the volume of infection. An unsupervised preprocessing pipeline is proposed to extract relevant clinical features and utilize this information to employ a pretrained ImageNet EfficientNetB5 model to extract discriminative features. Subsequently, a shallow feed-forward neural network is trained to classify the CT scans into three classes, namely COVID-19, Community-Acquired Pneumonia, and Normal. Through various ablation studies, we find that a domain-specific preprocessing pipeline has a significant positive impact on classification accuracy. The infection segmentation mask generated from the preprocessed pipeline performs better than state-of-the-art supervised semantic segmentation models. Further, the estimated infection severity score is observed to be well correlated with radiologists’ assessments. The results confirm the importance of domain-specific pre-processing for training machine learning algorithms.
---
## Prerequisite for using the repository

1. Install the neccessary packages required to run the code.   

	<html>
		<body>
			<p>pip install -r requirements.txt</p>
		</body>
	</html>

2. Download the datasets from this [link](https://drive.google.com/file/d/11xcGidVmFfW3XgGTLpndvpz-yGes0a3q/view?usp=sharing) and keep in the *Dataset* directory.
	
3. To reproduce the results, the lung masks need to generate and the required masks are given in this [link](https://drive.google.com/drive/folders/1sbIQIkkSnsO2cfVlUwxwYaR61PPbc2IM?usp=sharing). Download and keep the data in the *TempDir* folder.
	- To generate lung masks for each dataset, run the following code:   
		<html>
		<body>
			<p>bash runLungMask.sh</p>
		</body>
		</html>
		

4. The preprocessed images for the CT scan classification is given in this [link](https://drive.google.com/drive/folders/1sbIQIkkSnsO2cfVlUwxwYaR61PPbc2IM?usp=sharing). Download the data and keep in the *Preprocessed_Datasets* folder.

5. The trained model weights be found [here](https://drive.google.com/drive/folders/1xtHvsSU-qb5X8GnRxE6GQm-VEgHdbMD0?usp=sharing).

---

## Run the reporsitory

1. For infection segmentation, run the following codes:
	<html>
		<body>
			<p>	- run InfectionMask48Slices.m (results are shown in TABLE 2)
				- run InfectionMask638Slices.m (results are shown in TABLE 3)
				- InfectionMaskMosmed.m (results are shown in TABLE 4)</p>
		</body>
	</html>
	
2. For classification task, run the following codes:
	- run ClassificationSPGC.m (results are shown in TABLE 5)
	- run ClassificationMosmed.m (results are shown in TABLE 6)
	- run ClassificationLDCT.m (results are shown in TABLE 6)
	- run ClassificationLDCT_PCR.m (results are shown in TABLE 6)
3. Baseline model can be found [here](https://github.com/shubhamchaudhary2015/ct_covid19_cap_cnn).
4. For the chest CT severity score (CTSS), run the following code:
	- run 