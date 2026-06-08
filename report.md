# 

# 

# 

# Deep Learning for wildfire risk prediction in Spain

### Melker Svensson (242994), Felipe Lopez (266687) and Shuangjie Xia (269610)

### Group: *rho*

## ![][image1]

## Introduction

This project aims to support wildfire risk prediction in Spain by providing a more comprehensive evaluation of the IberFire dataset for baseline and deep learning models \[1\]. We will focus on predicting the probability of fire occurrence for the next day based on the previous week's data. The project will involve data sampling, cleaning, model training, and evaluation, with an emphasis on handling class imbalance and ensuring robust performance metrics.

## State of the art

In 2025, Ercibengoa et al. introduced IberFire \[1\], a spatio-temporal dataset at 1 km × 1 km × 1-day resolution covering mainland Spain from December 2007 to December 2024\. It integrates 260 features across eight categories: auxiliary location features, fire history, geography, land cover, topography, human activity, meteorological variables, and vegetation indices. For validation, the authors train a baseline model reported to achieve 0.955 AUC. However, the test set is drawn from a balanced subsample, which does not reflect real-world class imbalance where fire events represent a small fraction of all observations. Furthermore, using AUC as the single performance metric leaves the evaluation of false negatives unassessed. Finally, the addition of deep learning models such as Long Short-Term Memory Networks would enhance this validation by assessing their added value in this context.

## Methodology

### Data engineering

The data engineering phase consisted of three main stages: analysis, sampling and cleaning.

First, we analysed the IberFire dataset, consisting of spatio-temporal data of Spain across 17 years, with a daily frequency and a spatial resolution of 1km. The dataset includes a binary target variable "is\_fire" that indicates whether a fire occurred in a given cell on a given day, as well as various features such as land cover, population density, and weather variables. Figure A1 shows the values of “is\_fire” for a specific day across Spain. We found that the dataset is highly imbalanced, with only 0.002% of the samples being fire events. Due to this imbalance, and to keep data at a manageable size, we decided to sample a subset of IberFire for training our models.

The sampling strategy started by defining the training, validation and test splits. To avoid data leakage, we assigned the years 2008-2022 to the training set, 2023 to the validation set and 2024 to the test set. For the training set, we used a 1:3 ratio of fire to no-fire samples. For each fire sample, we generated three types of negative samples: spatial hard negatives (cells at least 5km away on the same date), temporal hard negatives (same cell and day but different year) and baseline negatives (random valid non-fire events). This approach was designed to create a more challenging training set that encourages the model to learn meaningful patterns rather than relying on trivial heuristics, such as making positive predictions for summer only (Figure A2). For the LSTM model, we included a 7-day window of data for each sample to capture temporal dependencies. For the validation and test sets, we sampled unbalanced negatives at a 1:10 ratio by drawing random valid non-fire events from the respective splits.

The raw IberFire dataset contains three vintages of CORINE Land Cover (CLC) data (2006, 2012, 2018), each providing two representations: raw numeric class columns (e.g. CLC\_2006\_1 through CLC\_2006\_44), and named aggregate columns (e.g. CLC\_2006\_urban\_fabric\_proportion) that already summarise those raw codes into 19 meaningful land-cover categories. Since the named variables are a complete and interpretable summary of the raw classes, the numeric columns are dropped. The three remaining year-sets of 19 named proportions are then collapsed into a single set of 19 columns by selecting the temporally appropriate vintage per row: observations before 2011 receive 2006 CLC values, those between 2012 and 2017 receive 2012 values, and those after 2018 receive 2018 values. The same nearest-year logic is applied to population density, which ships as 13 annual columns (2008–2020) and is collapsed to a single “popdens” feature. Together these two operations remove roughly 180 columns from the raw feature space.

Beyond land cover, several additional preprocessing steps are applied. Columns with zero variance after sampling (is\_spain, is\_waterbody, is\_sea) are dropped, as are exact duplicates: label (identical to is\_fire), float coordinate duplicates (x\_coordinate, y\_coordinate), and grid-index duplicates (x\_index, y\_index). The is\_near\_fire flag is excluded since our target is focused on predicting the probability of fire for a specific cell, and we cannot assume that the presence of a near fire will be provided at inference time. Month is derived from the time column and encoded as cyclical sine/cosine pair to preserve calendar continuity across year boundaries, and autonomous community is one-hot encoded. Null values are imputed with per-feature medians fitted on the training set and applied without modification to validation and test sets. For the LSTM pipeline, observations within each 7-day sequence window are additionally forward-filled before median imputation, preserving short-term temporal continuity. Finally, all numeric features are z-score standardised using a StandardScaler fitted on training data, critical for LSTM gradient stability and harmless for XGBoost. The resulting feature set, detailed in Table 1, totals approximately 92 columns across all groups.

![][image2]

Table 1

### Models

#### XGBoost baseline

As a baseline we train an XGBoost gradient boosted tree on the full flat feature set. The model is GPU-accelerated and uses binary logistic loss with “scale\_pos\_weight” set to the train negative-to-positive ratio to account for class imbalance. Early stopping is based on validation PR-AUC rather than log-loss, since PR-AUC is more informative for imbalanced problems where we care primarily about recall of the minority class. The baseline serves as a strong reference point given that gradient boosted trees are known to perform well on structured tabular data.

We also examine the effect of removing the fire weather index (FWI) feature. FWI is a physics-based index encoding cumulative atmospheric conditions. It is, in effect, a manually engineered temporal model. Its presence may allow both XGBoost to shortcut temporal reasoning: instead of learning how raw weather sequences evolve into dangerous conditions, the model can read the danger level directly from FWI. 

#### LSTM ablation

To find a good LSTM architecture we run a grid search over 27 configurations: hidden size ∈ {32, 64, 128}, number of layers ∈ {1, 2, 3}, and learning rate ∈ {1e-3, 3e-4, 1e-4}. All runs use BCE loss with a positive class weight and train sequentially on a single GPU job.

#### Focal loss variant

The best architecture from the ablation is retained with Focal Loss (α=0.75, γ=2.0) instead of BCE. The motivation is that BCE treats all misclassified examples equally, whereas fires are inherently hard examples. Many fire cells look similar to high-risk non-fire cells on any given day. The idea is that focal loss could down-weights easy negatives (cells the model is already confident are not fires) and concentrates the gradient signal on the uncertain, difficult cases. Early stopping is removed at this stage after observing that several ablation runs were cut off early on. Instead both models train for the full 50 epochs and we save two checkpoints: the one with the best validation PR-AUC and the one with the lowest validation false negative count.

#### Dropout variant

We also try adding dropout (rates 0.2, 0.3, 0.35) to the best architecture as a regularisation measure. From the training curves it was clear the LSTM peaks in the first couple of epochs and then starts to overfit, so dropout was a natural response to that. It is applied both between LSTM layers and on the final hidden state before the classifier. The same dual-checkpoint strategy is used.

Best ablation without FWI

The same reasoning goes for LSTM as with the baseline XGboost. On top of that, removing this temporal variable forces the LSTM to discover these temporal dependencies from raw inputs. This provides a cleaner test of whether the LSTM adds anything beyond what the static features already encode.  

#### Ensemble

Finally, we combine the frozen pre-trained XGBoost with the best LSTM variant encountered, *lstm-h128-l2-lrle-4-without-FWI*, using a simple 50/50 score average. Neither the XGBoost nor the LSTM weights are updated; their output probabilities are simply averaged, and the optimal classification threshold is then swept on the validation set. The pairing is intentional: XGBoost retains the FWI feature, giving it access to accumulated fire weather danger, while the LSTM operates on raw weather sequences without FWI, having been forced to learn temporal dynamics directly. The hope is that the two models capture complementary signals, and that their combination produces a more robust risk score than either model alone. 

## Experiments 

Since the primary objective is to minimize missed fire events, all models are evaluated with a focus on recall rather than overall accuracy. A false negative (i.e. predicting no fire when one occurs), carries a much higher operational cost than a false positive, so standard accuracy or ROC-AUC alone are insufficient evaluation criteria. The main ranking metric used throughout is PR-AUC (area under the precision-recall curve), which is threshold-independent and more informative for FN/FP. The five model families trained are shown in Table 2\.

![][image3]

Table 2

## Results

### XGBoost baseline

From looking at the validation curve (Figure 1\) we see a positive evolution where the model improves noticeably during the first steps and then successively plateaus. As mentioned before, we consider it more precise to look at the PR-AUC and FN numbers than ROC-AUC. When comparing these numbers at Table 1 for the “no\_fwi” and “with\_fwi” we see a slight improvement for the latter across all metric scores. 

![][image4]

Table 1 

![][image5]

Figure 1 

### LSTM ablation

When looking at Figure 2, we can see that none of the LSTM architectures from the ablation study perform better than the baseline on their respective best threshold. They all have a smaller PR-AUC. The best configuration from the ablation study is the “lstm-h128-l2-lrle-4”, referring to a hidden state of 128 nodes, with 2 layers and learning rate 10\-3. From now on we will refer to this setup as “best\_from\_ablation”.

![][image6]

Figure 2\. For interpretability purposes we only include the best performing architectures from the ablation study.

### Focal loss and dropout variants

From looking at Figure 3 we see similar behaviours to the ablation study: the models improve quickly in a few epochs and then start overfitting on the training data leading to worse performance on the validation set. By looking at the highest peaks we see that the “best\_from\_ablation”, “lstm-focal-loss” and “lstm-dropout-0.2” perform very similarly with no real improvement. It is important to highlight that for the ablation study we included “early stopping” which is why the green curve is shorter than the others.

### 

###  ![][image7]

Figure 3

### FWI comparison

Looking at Table 2, the configuration without FWI consistently outperforms the one with FWI across all reported metrics: higher PR-AUC (0.787 vs 0.757), better precision (0.585 vs 0.553), lower FN rate (0.174 vs 0.180), and notably fewer false positives (4034 vs 4568).

### ![][image8]

Table 3

### Ensemble and Comparison table

From looking at Table 3, the “Baseline” model achieves the highest recall (0.853) and lowest FN rate (0.147) of all individual models, outperforming “Best\_from\_ablation with/without FWI”, “Best\_focal”, and “Best\_dropout” variants across most metrics despite its notably high FP count (5689). The “Ensemble” is the clear winner overall, it leads on PR-AUC (0.845), precision (0.606), and FP count (3816) while maintaining competitive recall and FN rate, striking the best balance between catching fires and avoiding false alarms.

![][image9]

Table 4

## Discussion

### Why the LSTM could not overcome the baseline?

Our initial motivation for implementing an LSTM was that we expected time-varying features to be powerful for the model. For example, if the temperature has steadily been increasing over the last days, then the model would flag this as risky, and potentially correctly predict fire.

This did not end up happening and the most likely explanation is that the dominant signal in this dataset is not sequential but tabular. The static and slowly-varying features, such as forest proportion, vegetation, and geographic location, encode where fires tend to happen, and gradient boosted trees are specifically well-suited to learning these kinds of non-linear static feature interactions. Figures 4 and 5 show that the most influential features in both architectures are static. On top of this, the 7-day sequence window is short relative to fire-relevant timescales: soil moisture depletion and drought accumulation operate over weeks to months, well outside the model's view.

![][image10]![][image11]Figures 4 & 5 \- Feature importance for the LSTM (left) and XGBoost (right)

### Why does the LSTM perform better without FWI?

As mentioned before, FWI is a temporal variable with memory of prior weather. When fed into an LSTM alongside a raw 7-day weather window, we suspect the model is using the FWI value rather than learning the temporal weather data. Removing FWI forces the model to do the work it was architecturally designed for: extracting temporal patterns from raw weather sequences, sharpening gradients through the recurrent connections, and building a more meaningful internal representation of fire risk over time.

### Why does the ensemble achieve better performance?

At inference, the ensemble combines the predicted probabilities of both models into a single score (0.5 × XGBoost \+ 0.5 × LSTM). From the improvement in PR-AUC we can deduce that the two models have complementary strengths to a certain degree, potentially because XGBoost captures static tabular patterns well while the LSTM processes temporal sequences. Additionally, the decision threshold is re-optimised on this combined score based on the training set.

### The problem of binary classification

Fire occurrence is not just a function of conditions, it also requires an ignition event, which is largely random. A region can sustain extreme fire weather for three weeks and only burn on day 18 because someone throws a cigarette. A model trained on binary labels gets penalised for not predicting fire on days 1 through 17 despite those days being genuinely dangerous. This is not a modelling failure, it is a label problem. The result is that many false negatives are not actual misses but suppressed moderate-risk signals.

The root limitation is the loss function. For example, with BCE and assuming a threshold of 0,5, it penalises a prediction of 0.499 and 0.1 identically on a fire cell, despite one being far closer to the correct answer. It provides no reward for moving in the right direction or for developing risk awareness, only for crossing a fixed threshold.

This suggests a natural extension: reframing the task as regression rather than binary classification, with FWI as the target label. We could also supplement this label by some indicator encoding whether a fire actually occurred under the observed FWI conditions. Under such a framework, the model could develop into a genuine risk assessor rather than a fire/no-fire predictor.

## Future work

Future work in the area can be directed in several ways: modifying the target variable, extending the observation window, and trying different architectures.

First, the training objective can be reframed. As long as the model is trained to predict binary fire occurrence, it has no way to learn that predicting high risk in a dangerous area that does not burn is actually correct behaviour, the label always says it was wrong. Moving toward a risk assessment framing requires a different kind of supervision: assigning elevated risk scores to cells near a fire on the same day (since conditions in neighbouring cells were likely also dangerous), or using temporal proximity to give partial credit to days immediately before a fire ignition. 

Another possible improvement is extending the input sequence from 7 to 30 days or more. Drought accumulation, soil moisture depletion and prolonged heat waves operate on timescales the current window simply cannot capture, and an increase in the amount of prior values can potentially improve model performance.

Finally, as we only focused on the LSTM architecture, future projects can evaluate the possibility of implementing different deep learning techniques, such as Convolutional Neural Networks or attention-based networks, aiming to enhance pattern recognition and improve the predictive power.

## Conclusions

This project has demonstrated that robust wildfire risk prediction requires a shift from standard binary classification metrics to those that prioritize operational safety. While we achieved performance parity with the IberFire baseline, our evaluation reveals that ROC-AUC is an insufficient metric for capturing performance on minority fire events. By prioritizing PR-AUC, we established a more rigorous standard for evaluating false negatives, which is the critical operational risk.

Our experiments indicate that XGBoost was superior to LSTM architectures, likely due to the short 7-day input window and the power of static features. However, when combining both models, the best performance is achieved.

Future research should prioritize expanding input sequence lengths to capture multi-week environmental trends and moving toward auxiliary supervision methods, such as assigning risk scores to provide a more actionable decision-support tool for forest management authorities.

## 

## References

\[1\] Erzibengoa, J., Gómez-Omella, M., & Goienetxea, I. (2025). IberFire \- a detailed creation of a spatio-temporal dataset for wildfire risk assessment in Spain. https://doi.org/10.48550/arXiv.2505.00837

## Appendix

![][image12]

Figure A1 \- Value of is\_fire in Spain in 15/06/2022

![][image13]

Figure A2 \- Fires per month in the IberFire dataset
