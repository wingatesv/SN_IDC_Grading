# Is Stain Normalisation Conducive in Histopathological grading of Invasive Ductal Carcinoma?
## Abstract
**Background and Objective**: There is debate over the impact of Stain Normalisation on the results of recent breast cancer histopathological studies. Some studies claim that Stain Normalisation has no effect on the classification outcomes, while others suggest that it improves them. To address this uncertainty, the aim of our study was to investigate the effectiveness of Stain Normalisation in breast cancer histopathological classification tasks using convolutional neural networks, focusing specifically on the Invasive Ductal Carcinoma grading task. 

**Methods**: Six conventional and deep learning-based Stain Normalisation techniques (Reinhard, Macenko, Structure-preserving Colour Normalisation, Adaptive Colour Deconvolution, StainGAN, and StainNet) were selected to evaluate their effectiveness in the classification task. Five templates were selected for the conventional techniques as the target image. Seven ImageNet pre-trained CNNs (EfficientNet-B0, EfficientNet-V2-B0, EfficientNet-V2-B0-21k, ResNet-V1-50j, ResNet-V2-50, MobileNet-V1, and MobileNet-V2) were used as feature extractors in our classifiers for the implementation of the model. 

**Results**: The p-value of 0.11 between the mean Balanced Accuracy of models trained with StainGAN-normalized (best-performing Stain Normalisation technique) images and models trained with non-normalised images indicated statistical insignificance between the two variables. This means that our null hypothesis was not rejected and our findings indicate that models trained with the non-normalised dataset are likely to be more effective in the grading task.

**Conclusion**: Our study successfully showed that Stain Normalisation does not have a significant impact on the Invasive Ductal Carcinoma grading task, contradicting the assumption that Stain Normalisation would improve classification outcomes.

## Methodology
![image](https://user-images.githubusercontent.com/56868536/212927412-06cbc49c-149b-4cd6-898c-42d4df097fe5.png)

Figure 1. The overall methodology of the study. (1) The Four Breast Cancer Grades (FBCG) dataset is formed by combining the 400X Benign Class from the BreaKHis dataset with the remaining classes from the BCHI dataset. (2) The implemented model is trained with train set (DTR) from base dataset (DB) using the Stratified Five-fold Cross-validation (SFFCV) to evaluate model stability. (3) The hyperparameters of the model are optimised until the model is stable across each fold. (4) The SFFCV process is repeated until the model is optimised. (5) Once the model performance is satisfied, (6) the FBCG datasets are stain-normalised with different techniques to form stain-normalised dataset (DSN, T). (7) Lastly, each DSN, T and DB is fed forward into the model to retrain, followed by (8) obtaining the final test results.

![image](https://user-images.githubusercontent.com/56868536/212928102-12cb3a98-85af-40a7-a391-b3aa98fa0424.png)

Figure 2. The structure of the IDC grading classifier: (a) input layer, (b) augmentation layers, (c) feature extractor (non-trainable), (d) dropout layer, (e) dense layer (trainable), and (f) output layer.
## Results

![image](https://user-images.githubusercontent.com/56868536/216745586-1bc2da0e-b5e8-42b3-b6d9-f7733d37ed5e.png)

Figure 3. The mean test BAC scores of the seven models across T with different conventional SN techniques from Table 7, Table 8, Table 9 and Table 10 in the Appendix. The ACD technique tops other techniques across all templates but failed to outperform the baseline result.

![image](https://user-images.githubusercontent.com/56868536/216745608-4f78b616-def4-47e1-8ca3-4fe9805656ee.png)

Figure 4. The test BAC scores of seven models trained with StainGAN-normalised, StainNet-normalised and non-normalised datasets. Although the results are comparable among the deep learning-based SN techniques, the mean BAC scores of the seven models trained in the StainGAN-normalised dataset achieve slightly higher than models trained in the StainNet-normalised dataset but lower than the baseline result.

![image](https://user-images.githubusercontent.com/56868536/216745627-ae59237d-8f6b-4665-a563-366a3f9fd8ec.png)

Figure 5. The mean test BAC scores of the seven models trained in six different stain-normalised and the non-normalised FBCG datasets. Among the six SN techniques, the StainGAN technique outperforms other SN technique. However, the baseline result tops the best SN results by 0.0112 score.

## Conclusion
In this investigation, we set out to address the question of the effectiveness of Stain Normalisation (SN) in the task of Invasive Ductal Carcinoma (IDC) grading. To accomplish this, we utilised seven pre-trained Convolutional Neural Network (CNN) models as feature extractors to classify the FBCG dataset into four grades (G0, G1, G2, and G3). The FBCG dataset was normalised using six techniques: Reinhard, Macenko, SPCN, ACD, StainGAN, and StainNet. We also selected five templates for the conventional SN techniques to investigate their impacts on each method. We obtained the p-value of 0.11 when comparing test mean BAC score between models trained with the StainGAN-normalised (best SN technique) image and models trained with the non-normalised images, indicating a statistically insignificant difference as the null hypothesis: “A CNN trained with a stain-normalised dataset has no effect on the IDC grading accuracy”, is failed to be rejected. This contradicts the widely held belief that SN is crucial for histopathological classification tasks. In a case where SN is required in the image pre-processing pipeline, we recommend StainGAN, StainNet and ACD techniques over Reinhard, Macenko and SPCN techniques attributed to their excellent performance relatively in normalising images.
Looking forward, we plan to extend our study to other IDC grading datasets and various CNN architectures [74]-[76] as well as further our study to examine the generalisability of the CNN model with respect to colour features in IDC. Additionally, we aim to explore the inconsistent effects of SN on different breast cancer histopathological classification tasks.

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
