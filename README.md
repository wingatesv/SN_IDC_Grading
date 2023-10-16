# Evaluating the Effectiveness of Stain Normalization Techniques in Automated Grading of Invasive Ductal Carcinoma Histopathological Images?
## Abstract
**Background and Objective**: Debates persist regarding the impact of Stain Normalization (SN) on recent breast cancer histopathological studies. While some studies propose no influence on classification outcomes, others argue for improvement. This study aims to assess the efficacy of SN in breast cancer histopathological classification, specifically focusing on Invasive Ductal Carcinoma (IDC) grading using Convolutional Neural Networks (CNNs). The null hypothesis asserts that SN has no effect on the accuracy of CNN-based IDC grading, while the alternative hypothesis suggests the contrary. 

**Methods**: We evaluated six SN techniques, with five templates selected as target images for the conventional SN techniques. We also utilized seven ImageNet pre-trained CNNs for IDC grading. The performance of models trained with and without SN was compared to discern the influence of SN on classification outcomes. 

**Results**: The analysis unveiled a p-value of 0.11, indicating no statistically significant difference in Balanced Accuracy Scores between models trained with StainGAN-normalized images, achieving a score of 0.9196 (the best-performing SN technique), and models trained with non-normalized images, which scored 0.9308. As a result, we did not reject the null hypothesis, indicating that we found no evidence to support a significant discrepancy in effectiveness between stain-normalized and non-normalized datasets for grading tasks. 

**Conclusion**: This study demonstrates that SN has a limited impact on IDC grading, challenging the assumption of performance enhancement through SN.

## Methodology
![image](https://user-images.githubusercontent.com/56868536/212927412-06cbc49c-149b-4cd6-898c-42d4df097fe5.png)

Figure 1. The overall methodology of the study. 1) The FBCG dataset is assembled by combining images from the 400X Benign class of the BreaKHis dataset and images from the BCHI dataset. 2) To evaluate model stability, the implemented model is trained with DTR from DB using the Stratified Five-fold Cross-validation (SFFCV). 3) The hyperparameters of the model are optimized until the model is stable across each fold. 4) The SFFCV process is repeated until the model is optimized. 5) Once satisfactory model performance is achieved, 6) the FBCG datasets undergo stain normalization using various techniques to form DSN, T. 7) Lastly, each DSN, T and DB is fed forward into the model to retrain, followed by 8) obtaining the final test results.

![image](https://user-images.githubusercontent.com/56868536/212928102-12cb3a98-85af-40a7-a391-b3aa98fa0424.png)

Figure 2. The structure of the IDC grading model: (a) input layer, (b) augmentation layers, (c) feature extractor (non-trainable), (d) dropout layer, (e) dense layer (trainable), and (f) output layer.
## Results

![image](https://user-images.githubusercontent.com/56868536/216745586-1bc2da0e-b5e8-42b3-b6d9-f7733d37ed5e.png)

Figure 3. The mean test BAC scores of the seven models across T with different conventional SN techniques from Table 7, Table 8, Table 9 and Table 10 in the Appendix. The ACD technique tops other techniques across all templates but failed to outperform the baseline result.

![image](https://user-images.githubusercontent.com/56868536/216745608-4f78b616-def4-47e1-8ca3-4fe9805656ee.png)

Figure 4. The test BAC scores of seven models trained with StainGAN-normalised, StainNet-normalised and non-normalised datasets. Although the results are comparable among the deep learning-based SN techniques, the mean BAC scores of the seven models trained in the StainGAN-normalised dataset achieve slightly higher than models trained in the StainNet-normalised dataset but lower than the baseline result.

![image](https://user-images.githubusercontent.com/56868536/216745627-ae59237d-8f6b-4665-a563-366a3f9fd8ec.png)

Figure 5. The mean test BAC scores of the seven models trained in six different stain-normalised and the non-normalised FBCG datasets. Among the six SN techniques, the StainGAN technique outperforms other SN technique. However, the baseline result tops the best SN results by 0.0112 score.

## Conclusion
In this study, we set out to address the question of the effectiveness of Stain Normalization (SN) in the task of Invasive Ductal Carcinoma (IDC) grading. To accomplish this, we utilized seven pre-trained Convolutional Neural Network (CNN) models as feature extractors to classify the FBCG dataset into four IDC grades. The FBCG dataset was stain-normalized using six techniques: Reinhard, Macenko, SPCN, ACD, StainGAN, and StainNet. For the conventional SN techniques, we selected five templates to investigate their impacts on each method. We conducted a comparative analysis of models trained with and without SN to understand the impact of SN on the classification results. Our findings revealed a p-value of 0.11 when comparing the test mean Balanced Accuracy (BAC) score of models trained with StainGAN-normalized (best-performing SN technique) images and non-normalized images. This indicates that there is no statistically significant difference in the effectiveness of stain-normalized and non-normalized datasets for IDC grading tasks. Contrary to common belief, our study suggests that SN may not be as crucial for histopathological classification tasks as previously thought. However, if SN is required in the image pre-processing pipeline, we recommend StainGAN, StainNet, and ACD techniques due to their relative performance in stain-normalizing images. Looking forward, in addition to extending our future work with the consideration mentioned in Section 4.5, we plan to 77–79examine the generalizability of the CNN model with respect to color features in IDC. Additionally, we aim to explore the inconsistent effects of SN on different breast cancer histopathological classification tasks.

## Acknowlegment and Special Thanks
Reinhard, Macenko and SPCN Techniques: 

P. Byfield, StainTools, (2020). https://github.com/Peter554/StainTools

ACD Techniques:

Zheng et al. Adaptive Color Deconvolution for Histological WSI Normalization, (2019). https://github.com/Zhengyushan/adaptive_color_deconvolution

StainGAN and StainNet Techniques:

Kang et al. StainNet: a fast and robust stain normalization network, (2022). https://github.com/khtao/StainNet

## Data Availability
The FBCG Dataset:

![Screenshot 2023-01-18 at 3 37 23 PM](https://user-images.githubusercontent.com/56868536/213111557-8ad6de49-639e-477d-a259-a359d556a6c1.png)

The original datasets used to form the Four Breast Cancer Grades (FBCG) Dataset:

BreaKHis Dataset: https://web.inf.ufpr.br/vri/databases/breast-cancerhistopathological-database-breakhis/

BCHI Dataset: https://zenodo.org/record/834910#.WXhxt4jrPcs.

## Cite this repository

If you find this repository useful or use it in your research, please consider citing our paper:
```
@article{
    XXX,
    author = {XXX},
    doi = {XXX},
    issn = {XXX},
    journal = {XXX},
    month = {XXX},
    number = {XXX},
    pages = {XXX},
    publisher = {XXX},
    title = {{XXX}},
    url = {XXX},
    volume = {XXX},
    year = {XXX}
}
```
# Pytorch Implementation
check this out for pytorch implementation: https://github.com/wingatesv/IDC_Grading_PyTorch.git
