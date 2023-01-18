# Is Stain Normalisation Effective in the Invasive Ductal Carcinoma Grading Task?
## Abstract
**Background and Objective**: Controversy has been discovered in the effect of Stain Normalisation among recent breast cancer histopathological studies. Several studies suggested that SN has no effect on the classification results, while others claimed that SN improves the classification outcomes. Therefore, we aim to investigate the effectiveness of Stain Normalisation in breast cancer histopathological classification tasks using convolutional neural networks, specifically in the Invasive Ductal Carcinoma grading task. 

**Methods**: We selected six types of conventional and deep learning-based Stain Normalisation techniques (Reinhard, Macenko, Structure-preserving Colour Normalisation, Adaptive Colour Deconvolution, StainGAN and StainNet) to study their effectiveness in the classification task. Furthermore, we selected five templates for the conventional techniques as the target image. For the model implementation, we transferred seven ImageNet pre-trained CNNs (EfficientNet-B0, EfficientNet-V2-B0, EfficientNet-V2-B0-21k, ResNet-V1-50j, ResNet-V2-50, MobileNet-V1, MobileNet-V2) as feature extractors in our classifiers to conduct the classification task. 

**Results**: We obtained a p-value of 0.11 between the mean Balanced Accuracy of models trained with the StainGAN-normalised (best-performing Stain Normalisation technique) images and models trained with the non-normalised images, indicating statistical insignificance between the two variables. Therefore, not only is the null hypothesis failed to be rejected, our findings indicate that models trained with the non-normalised dataset are likely to be more effective in the grading task. 

**Conclusion**: We successfully showed that Stain Normalisation does not significantly affect the Invasive Ductal Carcinoma grading task, opposing the presumption that Stain Normalisation will improve classification outcomes.

## Methodology
![image](https://user-images.githubusercontent.com/56868536/212927412-06cbc49c-149b-4cd6-898c-42d4df097fe5.png)

Figure 1. The overall methodology of the study. (1) The Four Breast Cancer Grades (FBCG) dataset is formed by combining the 400X Benign Class from the BreaKHis dataset with the remaining classes from the BCHI dataset. (2) The implemented model is trained with train set (DTR) from base dataset (DB) using the Stratified Five-fold Cross-validation (SFFCV) to evaluate model stability. (3) The hyperparameters of the model are optimised until the model is stable across each fold. (4) The SFFCV process is repeated until the model is optimised. (5) Once the model performance is satisfied, (6) the FBCG datasets are stain-normalised with different techniques to form stain-normalised dataset (DSN, T). (7) Lastly, each DSN, T and DB is fed forward into the model to retrain, followed by (8) obtaining the final test results.

![image](https://user-images.githubusercontent.com/56868536/212928102-12cb3a98-85af-40a7-a391-b3aa98fa0424.png)

Figure 2. The structure of the IDC grading classifier: (a) input layer, (b) augmentation layers, (c) feature extractor (non-trainable), (d) dropout layer, (e) dense layer (trainable), and (f) output layer.
## Results

![image](https://user-images.githubusercontent.com/56868536/212928361-90a8cb8c-59cd-43d1-92e4-4014d1103c93.png)

Figure 3. The mean test Balanced Accuracy (BAC) scores of the seven models across T with different conventional SN techniques. The ACD technique tops other techniques across all templates but failed to outperform the baseline result.


![image](https://user-images.githubusercontent.com/56868536/212928473-dd7fb288-e9d2-485e-bb50-a9b405c7f754.png)

Figure 4. The test BAC scores of seven models trained with StainGAN-normalised and StainNet-normalised FBCG datasets. Although the results are comparable, the mean BAC scores of the seven models trained in the StainGAN-normalised dataset achieve higher than models trained in the StainNet-normalised dataset but lower than the baseline result.

![image](https://user-images.githubusercontent.com/56868536/212928536-3a6835e2-200e-4091-b2fd-e859ccc3ccc4.png)

Figure 5. The mean test BAC scores of the seven models trained in six different stain-normalised and the non-normalised FBCG datasets. Among the six SN techniques, the StainGAN technique outperforms other SN technique. However, the baseline result tops all SN results by 0.0112 score.

## Conclusion
In this study, we attempted to answer the question: "Is SN effective in the IDC grading task?". We selected seven pre-trained CNN architectures as feature extractors to classify the FBCG dataset into four classes: (1) G0, (2) G1, (3) G2, and (4) G3. We normalised the FBCG dataset using six techniques (Reinhard, Macenko, SPCN, ACD, StainGAN and StainNet). We selected five templates for the conventional SN techniques to investigate their impacts on each method. We obtained the p-value of 0.11 when comparing test mean BAC score between models trained with the StainGAN-normalised (best SN technique) image and models trained with the non-normalised images, indicating statistically insignificant. Therefore, the H0: A CNN trained with a stain-normalised dataset has no effect on the IDC grading accuracy, is failed to be rejected. Our findings contradict the general presumption that SN is essential to perform well in histopathological classification tasks. In a case where SN is required in the data pre-processing pipeline, we recommend StainGAN, StainNet and ACD techniques over Reinhard, Macenko and SPCN techniques attributed to their excellent performance relatively in normalising images.
In future, we intend to repeat our study with other IDC grading datasets [53], [72] along with different state-of-the-art CNN architectures [74]–[76]. We aim investigate the impact of colour features in IDC on the generalisability of the CNN model. Additionally, we would like to investigate the inconsistent effect of SN on various breast cancer histopathological classification tasks. 

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
