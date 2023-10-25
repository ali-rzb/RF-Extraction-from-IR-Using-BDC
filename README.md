# Non-Contact Respiratory Flow Extraction from Infrared Images Using Balanced Data Classification
### This code is for my accepted paper in ICONIP 2023 conference.
The COVID-19 pandemic has emphasized the need for non-contact ways of measuring vital signs. However, collecting respiratory signals can be challenging due to the transmission risk and physical discomfort of spirometry devices. This is problematic in places like schools and workplaces where monitoring health is crucial. Infrared fever meters are not accurate enough since fever is not the only symptom of these diseases. The objective of our study was to develop a non-contact method for obtaining Respiratory Flow (RF) from infrared images. We recorded infrared images of three subjects at a distance of 1 meter while they breathed through a spirometry device. We proposed a method called Balanced Data Classification to distribute frames equally into several classes and then used the DenseNet-121 Convolutional Neural Network Model to predict RF signals from the infrared images. Our results showed a high correlation of 97% and a RMSE of 5%, which are significant compared to other studies. Our method is fully non-contact and involves standing at a distance of 1 meter from the subjects. In conclusion, our study demonstrates the feasibility of using infrared images to extract RF. 
## Overal
![Picture1](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/5a7eff2f-f403-469e-be52-8e938d7931fe)

## Experimental Setup
![Picture2](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/96410b56-901e-4d2d-b67f-72c7fbd22aa4)

## ROI Extraction
![first_point_article_new](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/f94fd56e-c5c3-4ed4-a943-e7c608cb2acd)

## Preprocessing
![Picture1](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/428da52b-6c2b-48fe-99a2-5e9bb775d173)

## Data Classification
# Balanced Data Classification (BDC)
BDC is a strategy that refines classification boundaries to foster a semi-uniform distribution across classes, as illustrated in below : 
![layers-COMPRESS](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/6802a519-0549-4be5-962c-6b4ec0556a01)

As depicted in Figure below, the BDC method contracts the boundaries for classes with higher populations, resulting in a more equitable data distribution.
![hist_norm_vs_ddl](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/020cdae4-5126-4a17-a070-62d58f066c7c)


# Results
## Finding best CNN
![Screenshot 2023-10-25 115043](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/25cca865-bb49-42cf-917a-c610b6e1c3a1)
## Results for DenseNet121
![Screenshot 2023-10-25 115151](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/0caaa793-6e9e-4f44-b712-c45e59fd2588)

## Prediction results

![Pred_C10_00_0_pp](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/3cee5cb5-c947-4763-abbb-ca55463a1b88)
![Pred_C10_01_0_pp](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/c48b67cd-b042-48ea-a506-b2e932f3228c)
![Pred_C10_02_0_pp](https://github.com/ali-rzb/RF-Extraction-from-IR-Using-BDC/assets/63366614/51dc2b5d-307a-4882-9fd0-3bf04c25f54f)

### Metrics
Metric     | Test 1    | Test 2    | Test 3    | Avg       |    
---------- | --------- | --------- | --------- | --------- |
F1 (%)     | 95.4194   | 88.1381   | 95.256    | 92.9378   |
Corr (%)   | 95.9102   | 97.7507   | 98.5779   | 97.4129   |
R² (%)     | 89.2655   | 95.3675   | 97.1347   | 93.9225   |
MAE (%)    | 0.638     | 2.5947    | 3.7602    | 2.3309    |
MSE        | 0.001     | 0.0063    | 0.0015    | 0.0029    |
RMSE       | 0.0318    | 0.0793    | 0.0391    | 0.05      |
