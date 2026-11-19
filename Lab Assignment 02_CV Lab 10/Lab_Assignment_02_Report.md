# Lab Assignment 02: Comparative Implementation of Machine Learning Regression Algorithms

**COMSATS University Islamabad, Wah Campus**  
**Name:** Muhammad Umer Rafiq (FA23-BAI-013)  
**Subject:** Introduction to Computer Vision  
**Instructor:** Dr. Jamal Hussain Shah  
**Date:** 19-Nov-2026

---

## 1. Objective

In this lab I had to solve one prediction problem with six regression algorithms and then compare them. The algorithms are Linear Regression, Decision Tree, Random Forest, Neural Network, XGBoost and K-Nearest Neighbors (KNN). For each algorithm I did two things. First I explained how it works, wrote its equations and solved a small example by hand. Then I implemented it in Python on the full dataset.

All six models use the same dataset, the same input feature, the same target and the same 80:20 train/test split, so the comparison is fair. The models are checked with MAE, MSE, RMSE and R². At the end I compare the results, look at the graphs and decide which algorithm is the best.

## 2. Environment

The code was run in Google Colab on a Tesla T4 GPU (the notebook printed: Device: cuda | Tesla T4). I used pandas and numpy for data, matplotlib for graphs, scikit-learn for the models and metrics, XGBoost for gradient boosting and PyTorch for the neural network. The neural network and XGBoost were trained on the GPU. The complete code is in the notebook Regression_Algorithms_Comparison.ipynb.

## 3. Dataset

I used the Salary Dataset (Simple Linear Regression) from Kaggle, uploaded by abhishek14398. It was downloaded with kagglehub and the files were stored at /kaggle/input/salary-dataset-simple-linear-regression. After removing the extra index column the data has 30 rows and 2 columns. The input feature is YearsExperience and the target is Salary.

*Table 1: First five rows of the dataset*

|  | YearsExperience | Salary |
| :---: | ---: | ---: |
| 0 | 1.2 | 39,344 |
| 1 | 1.4 | 46,206 |
| 2 | 1.6 | 37,732 |
| 3 | 2.1 | 43,526 |
| 4 | 2.3 | 39,892 |

*Table 2: Summary statistics of the dataset*

|  | YearsExperience | Salary |
| :--- | ---: | ---: |
| count | 30 | 30 |
| mean | 5.4133 | 76,004.00 |
| std | 2.8379 | 27,414.43 |
| min | 1.2000 | 37,732.00 |
| 25% | 3.3000 | 56,721.75 |
| 50% | 4.8000 | 65,238.00 |
| 75% | 7.8000 | 100,545.75 |
| max | 10.6000 | 122,392.00 |

The check for missing values gave 0 for both columns and there were 0 duplicate rows, so no cleaning was needed. Employees in the data have between 1.2 and 10.6 years of experience and salaries between 37,732 and 122,392. The Pearson correlation between experience and salary is 0.9782, which is very close to 1.

![Figure 1: Salary vs experience (scatter plot) and the distribution of both columns](images/cell5_1.png)

*Figure 1: Salary vs experience (scatter plot) and the distribution of both columns*

In the scatter plot the points go up in an almost straight line, so salary increases steadily with experience. This tells us the relationship is nearly linear, and I kept this in mind when looking at the results later. The two histograms show that most experience values are spread between 1 and 10 years and the salary values are spread between about 40,000 and 120,000.

## 4. Preprocessing and Train/Test Split

The only preprocessing step needed was dropping the unnamed index column. The data was then split 80:20 with random_state = 42, which gave 24 rows for training and 6 rows for testing (the output was: Train: 24 rows | Test: 6 rows). The same split is used for every model.

Feature scaling was done with StandardScaler for Linear Regression, KNN and the Neural Network. The scaler was fitted on the training data only and then applied to the test data, so no test information leaks into training. Decision Tree, Random Forest and XGBoost only compare values to thresholds, so they do not need scaling and were given the original values.

$$
z = \frac{x - \mu}{\sigma}
$$

For the models that have settings (Decision Tree, Random Forest, KNN and XGBoost) I used GridSearchCV with 5-fold cross-validation on the training set only. The test set was never used for tuning. The best settings found are given in Table 3.

*Table 3: Best hyper-parameters found by GridSearchCV*

| Model | Best parameters |
| :--- | :--- |
| Linear Regression | No tuning needed (StandardScaler + LinearRegression) |
| Decision Tree | max_depth = None, min_samples_leaf = 1 |
| Random Forest | n_estimators = 300, max_depth = 3, min_samples_leaf = 1 |
| Neural Network | 1-64-32-1, ReLU, Adam (lr = 0.01, weight decay = 0.0001), 1500 epochs |
| XGBoost | n_estimators = 100, max_depth = 2, learning_rate = 0.05 |
| KNN | n_neighbors = 6, weights = distance |

## 5. Evaluation Metrics

Let yᵢ be the actual value, ŷᵢ the predicted value, ȳ the mean of the actual values and n the number of test samples. The four metrics are:

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
$$

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

$$
RMSE = \sqrt{MSE}
$$

$$
R^2 = 1-\frac{\sum (y_i-\hat y_i)^2}{\sum (y_i-\bar y)^2}
$$

MAE is the average size of the error in salary units. MSE squares the errors, so big mistakes count more. RMSE is the square root of MSE and is again in salary units. R² tells how much of the change in salary the model explains, and 1 is the best possible value. For MAE, MSE and RMSE a smaller value is better.

## 6. Part A: Handwritten Implementation

For the handwritten part I took a small sample of 6 rows from the training set. The same 6 rows are used for all six algorithms. The point to be predicted is the first row of the test set, which has x = 9.7 years of experience and an actual salary of 112,636.

*Table 4: Sample of 6 training rows used for the hand calculations*

| i | x (years) | y (salary) |
| :---: | ---: | ---: |
| 1 | 1.6 | 37,732 |
| 2 | 2.1 | 43,526 |
| 3 | 4.1 | 56,958 |
| 4 | 4.6 | 61,112 |
| 5 | 5.2 | 66,030 |
| 6 | 8.8 | 109,432 |

### 6.1 Linear Regression

Working procedure: Linear Regression fits a straight line ŷ = β₀ + β₁x through the data. The slope and intercept are chosen so that the sum of squared differences between actual and predicted values is as small as possible (ordinary least squares). First the means x̄ and ȳ are found, then the slope β₁ is calculated from the deviations, and the intercept β₀ is found from the means.

$$
\hat y=\beta_0+\beta_1x
$$

$$
\beta_1=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2},\qquad \beta_0=\bar y-\beta_1\bar x
$$

Solution: x̄ = (1.6 + 2.1 + 4.1 + 4.6 + 5.2 + 8.8) / 6 = 4.4 and ȳ = 62,465. The deviations and their products are in Table 5.

*Table 5: Linear Regression calculation on the sample*

| x | y | x − x̄ | y − ȳ | (x − x̄)(y − ȳ) | (x − x̄)² |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.6 | 37,732 | -2.8 | -24,733 | 69,252.4 | 7.84 |
| 2.1 | 43,526 | -2.3 | -18,939 | 43,559.7 | 5.29 |
| 4.1 | 56,958 | -0.3 | -5,507 | 1,652.1 | 0.09 |
| 4.6 | 61,112 | 0.2 | -1,353 | -270.6 | 0.04 |
| 5.2 | 66,030 | 0.8 | 3,565 | 2,852.0 | 0.64 |
| 8.8 | 109,432 | 4.4 | 46,967 | 206,654.8 | 19.36 |
|  |  |  | Sum | 323,700.4 | 33.26 |

So Sxy = 323,700.40 and Sxx = 33.26. Then β₁ = 323,700.40 / 33.26 = 9,732.42 and β₀ = 62,465 − 9,732.42 × 4.4 = 19,642.34. The fitted line is ŷ = 19,642.34 + 9,732.42x. For the test point x = 9.7:

$$
\hat y = 19{,}642.34 + 9{,}732.42\times 9.7 = 114{,}046.84
$$

The actual salary is 112,636, so the hand calculation is off by about 1,411. I also checked it with the normal equation β = (XᵀX)⁻¹Xᵀy and with scikit-learn's LinearRegression. Both gave the same values (intercept 19,642.34 and slope 9,732.42), so the hand calculation is correct. Every salary goes up by about 9,732 for each extra year of experience.

### 6.2 Decision Tree Regression

Working procedure: A decision tree splits the data again and again using questions of the form x ≤ t. For every possible threshold t (the midpoints between neighbouring x values) the data is divided into a left and a right part, and the sum of squared errors (SSE) of both parts is added. The threshold with the smallest total SSE is chosen. This is repeated inside each part until the maximum depth is reached or a part has only one point. The prediction of a leaf is the mean of the salaries in that leaf.

$$
SSE(t)=\sum_{i\in L}(y_i-\bar y_L)^2+\sum_{i\in R}(y_i-\bar y_R)^2
$$

$$
\hat y_{leaf}=\frac{1}{n_{leaf}}\sum_{i\in leaf}y_i
$$

Solution: Before any split the mean of the 6 salaries is 62,465 and the SSE is 3,221,172,982. The 5 candidate thresholds are tried in Table 6.

*Table 6: Candidate splits for the first split*

| Threshold | nL | mean L | nR | mean R | SSE L | SSE R | SSE total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.85 | 1 | 37,732.0 | 5 | 67,411.6 | 0 | 2,487,107,435 | 2,487,107,435 |
| 3.10 | 2 | 40,629.0 | 4 | 73,383.0 | 16,785,218 | 1,773,955,076 | 1,790,740,294 |
| 4.35 | 3 | 46,072.0 | 3 | 78,858.0 | 194,542,712 | 1,414,247,576 | 1,608,790,288 |
| 4.90 | 4 | 49,832.0 | 2 | 87,731.0 | 364,193,912 | 941,866,802 | 1,306,060,714 |
| 7.00 | 5 | 53,071.6 | 1 | 109,432.0 | 574,094,075 | 0 | 574,094,075 |

The smallest total SSE is 574,094,075 at the threshold x ≤ 7.0, so this becomes the first split (SSE falls from 3,221,172,982 to 574,094,075). The right part has only one point, so it becomes a leaf. The left part (5 points) is split again in the same way and the best threshold is x ≤ 3.1. With a maximum depth of 2 the tree is:

```text
if x <= 7.0:
    if x <= 3.1:
        predict 40,629.00
    else:
        predict 61,366.67
else:
    predict 109,432.00
```

For x = 9.7 the answer is x > 7.0, so the prediction is 109,432. The actual value is 112,636, so the error is 3,204. The scikit-learn DecisionTreeRegressor with max_depth = 2 also gave 109,432.

### 6.3 Random Forest Regression

Working procedure: A random forest builds many decision trees. Each tree is trained on a bootstrap sample, which means rows are picked randomly with replacement from the training data, so some rows appear twice and some do not appear at all. The final prediction is the average of all the trees. Averaging reduces the variance of a single tree. Since there is only one feature, no random feature selection is needed here.

$$
\hat y(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
$$

Solution: I used B = 3 trees of depth 2 on the 6 sample rows. The bootstrap samples (row numbers from Table 4) and the trees are shown below.

*Table 7: Bootstrap samples and tree predictions for x = 9.7*

| Tree | Rows picked (i) | Prediction |
| :--- | :--- | ---: |
| Tree 1 | 1, 1, 3, 4, 5, 6 | 109,432.00 |
| Tree 2 | 2, 2, 3, 5, 6, 6 | 109,432.00 |
| Tree 3 | 2, 2, 3, 4, 4, 5 | 66,030.00 |

```text
Tree 1:
  if x <= 7.0:
      if x <= 1.6: predict 37,732.00
      else:        predict 61,366.67
  else:            predict 109,432.00
Tree 2:
  if x <= 7.0:
      if x <= 2.1: predict 43,526.00
      else:        predict 61,494.00
  else:            predict 109,432.00
Tree 3:
  if x <= 2.1:     predict 43,526.00
  else:
      if x <= 4.6: predict 59,727.33
      else:        predict 66,030.00
```

$$
\hat y = \frac{109{,}432 + 109{,}432 + 66{,}030}{3} = 94{,}964.67
$$

Trees 1 and 2 contain row 6 (x = 8.8) so they predict 109,432 for x = 9.7. Tree 3 did not pick row 6 at all, so the largest salary it knows is 66,030 and it predicts that value, which is far too low. The forest average is 94,964.67 against an actual value of 112,636. This shows that a forest with only 3 trees is not stable. The real forest in Part B uses 300 trees.

### 6.4 Neural Network Regression

Working procedure: I used a small network with 1 input, 2 hidden neurons with ReLU and 1 output. In the forward pass each hidden neuron computes zⱼ = w₁ⱼx + b₁ⱼ and hⱼ = max(0, zⱼ), and the output is ŷ = w₂₁h₁ + w₂₂h₂ + b₂. The loss is the mean squared error. In the backward pass the gradients are found with the chain rule and the weights are moved a small step against the gradient. The inputs and salaries are standardised first, using the mean and standard deviation of the 6 sample rows.

$$
z_j=w_{1j}x+b_{1j},\quad h_j=\max(0,z_j),\quad \hat y=\sum_j w_{2j}h_j+b_2
$$

$$
L=\frac{1}{n}\sum(\hat y_i-y_i)^2,\qquad \theta \leftarrow \theta-\eta\frac{\partial L}{\partial \theta}
$$

$$
\frac{\partial L}{\partial \hat y_i}=\frac{2}{n}(\hat y_i-y_i),\quad \frac{\partial L}{\partial w_{2j}}=\sum_i\frac{\partial L}{\partial \hat y_i}h_{ij},\quad \frac{\partial L}{\partial w_{1j}}=\sum_i\frac{\partial L}{\partial \hat y_i}w_{2j}\mathbb{1}[z_{ij}>0]\,x_i
$$

Solution: The sample has x̄ = 4.4, standard deviation of x = 2.3544, ȳ = 62,465 and standard deviation of y = 23,170.29. The starting weights were chosen by me as w₁ = [0.5, 0.8], b₁ = [0.1, -0.1], w₂ = [0.6, 0.4], b₂ = 0.05, and the learning rate is η = 0.1. Table 8 shows the forward pass.

*Table 8: Forward pass of the 1-2-1 network (standardised values)*

| x (std) | y (std) | z₁ | z₂ | h₁ | h₂ | ŷ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -1.189 | -1.067 | -0.495 | -1.051 | 0.000 | 0.000 | 0.050 |
| -0.977 | -0.817 | -0.388 | -0.882 | 0.000 | 0.000 | 0.050 |
| -0.127 | -0.238 | 0.036 | -0.202 | 0.036 | 0.000 | 0.072 |
| 0.085 | -0.058 | 0.142 | -0.032 | 0.142 | 0.000 | 0.135 |
| 0.340 | 0.154 | 0.270 | 0.172 | 0.270 | 0.172 | 0.281 |
| 1.869 | 2.027 | 1.034 | 1.395 | 1.034 | 1.395 | 1.229 |

The loss before the update is 0.4646. For the first two rows both z values are negative, so ReLU makes h = 0 and the network just outputs the bias 0.05 for them. After doing the backward pass by hand I got the gradients in Table 9 and then updated each weight with θ − 0.1 × gradient.

*Table 9: Gradients and one gradient-descent update*

| Parameter | Old value | Gradient | New value |
| :--- | ---: | ---: | ---: |
| w₁₁ | 0.5000 | -0.2944 | 0.5294 |
| w₁₂ | 0.8000 | -0.1932 | 0.8193 |
| b₁₁ | 0.1000 | -0.0336 | 0.1034 |
| b₁₂ | -0.1000 | -0.0895 | -0.0910 |
| w₂₁ | 0.6000 | -0.2509 | 0.6251 |
| w₂₂ | 0.4000 | -0.3640 | 0.4364 |
| b₂ | 0.0500 | 0.6055 | -0.0106 |

After this one step the loss went down from 0.4646 to 0.3977, so the update moved in the right direction. I verified the gradients with PyTorch autograd and got the same numbers for dW₂ and dW₁. For x = 9.7 the network after one step predicts 98,707.68 (after converting back from the standardised scale), compared with the actual 112,636. One step is not enough for a good prediction. In the Python implementation the same idea is repeated for 1500 epochs.

### 6.5 XGBoost Regression

Working procedure: XGBoost builds trees one after another and each new tree tries to correct the errors of the previous ones. It starts with the mean of y as the first prediction. In every round it calculates the gradient g and hessian h of the squared error loss for each row, chooses the split with the highest Gain, calculates the leaf weights and adds the tree to the prediction after multiplying by the learning rate η. λ is a regularisation term and γ is the minimum gain needed for a split.

$$
g_i=\hat y_i-y_i,\qquad h_i=1
$$

$$
Gain=\frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma
$$

$$
w=-\frac{G}{H+\lambda},\qquad \hat y^{(m)}=\hat y^{(m-1)}+\eta\,w
$$

Solution: I used λ = 1, γ = 0, η = 0.3, trees with one split and 2 rounds. The starting prediction is ȳ = 62,465 for every row. In round 1 the gradients are g = ŷ − y, and the Gain for each threshold is in Table 10.

*Table 10: Round 1 split search*

| Threshold | G_L | H_L | G_R | H_R | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.85 | 24,733 | 1 | -24,733 | 5 | 203,907,096 |
| 3.10 | 43,672 | 2 | -43,672 | 4 | 508,598,289 |
| 4.35 | 49,179 | 3 | -49,179 | 3 | 604,643,510 |
| 4.90 | 50,532 | 4 | -50,532 | 2 | 680,928,806 |
| 7.00 | 46,967 | 5 | -46,967 | 1 | 735,299,696 |

The best split is x ≤ 7.0. The leaf weights are w_L = −46,967 / (5 + 1) = −7,827.83 and w_R = 46,967 / (1 + 1) = 23,483.50. After multiplying by η = 0.3 the predictions for the 6 rows become 60,117 (first five rows) and 69,510 (last row). Round 2 is done the same way with the new gradients, shown in Table 11.

*Table 11: Round 2 split search*

| Threshold | G_L | H_L | G_R | H_R | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.85 | 22,384.65 | 1 | -27,081.35 | 5 | 184,809,123 |
| 3.10 | 38,975.30 | 2 | -43,672.00 | 4 | 442,327,718 |
| 4.35 | 42,133.95 | 3 | -46,830.65 | 3 | 494,471,798 |
| 4.90 | 41,138.60 | 4 | -45,835.30 | 2 | 517,808,586 |
| 7.00 | 35,225.25 | 5 | -39,921.95 | 1 | 500,266,401 |

This time the best split is x ≤ 4.9 (Gain 517,808,586), with w_L = −8,227.72 and w_R = 15,278.43. The predictions after round 2 are 57,648 for the first four rows, 64,700 for row 5 and 74,094 for row 6. For x = 9.7 the final prediction after 2 rounds is 74,093.58, which is far from the actual 112,636. This is expected because only 2 rounds with a learning rate of 0.3 were done, and the model had not finished moving towards the real values. A full XGBoost model uses many more trees.

### 6.6 K-Nearest Neighbors Regression

Working procedure: KNN does not train anything. For a new point it measures the distance to every training point, takes the k closest points and predicts the average of their targets. With one feature the distance is just the absolute difference.

$$
d(x_q,x_i)=\sqrt{\sum_j (x_{qj}-x_{ij})^2},\qquad \hat y(x_q)=\frac{1}{k}\sum_{i\in N_k(x_q)}y_i
$$

Solution: I took k = 3 and the query point x_q = 9.7. The distances are in Table 12.

*Table 12: Distances from x = 9.7 (sorted by rank)*

| Rank | i | x | y | Distance \|9.7 − x\| |
| :---: | :---: | ---: | ---: | ---: |
| 1 | 6 | 8.8 | 109,432 | 0.9 |
| 2 | 5 | 5.2 | 66,030 | 4.5 |
| 3 | 4 | 4.6 | 61,112 | 5.1 |
| 4 | 3 | 4.1 | 56,958 | 5.6 |
| 5 | 2 | 2.1 | 43,526 | 7.6 |
| 6 | 1 | 1.6 | 37,732 | 8.1 |

$$
\hat y=\frac{109{,}432+66{,}030+61{,}112}{3}=78{,}858
$$

The prediction is 78,858 against an actual value of 112,636. The reason is that in this small sample there is only one point close to 9.7. The other two neighbours (x = 5.2 and x = 4.6) are far away and have much lower salaries, which pulls the average down. KNN needs enough points around the query to work well.

## 7. Part B: Python Implementation

In this part every model was trained on the 24 training rows and tested on the 6 test rows. The split, the feature and the target are the same for all models. Linear Regression and KNN were used inside a pipeline with StandardScaler. The Neural Network is a PyTorch model with layers 1 → 64 → 32 → 1 and ReLU activations. It was trained with the Adam optimiser and MSE loss for 1500 epochs on the GPU, with a cosine learning-rate schedule, using full-batch training because the data is so small. XGBoost was run with device = cuda.

![Figure 2: Training loss of the neural network (log scale)](images/cell26_2.png)

*Figure 2: Training loss of the neural network (log scale)*

In Figure 2 the loss starts near 1 and drops very fast in the first 200 epochs. After that it goes down slowly and ends below 0.01. There are a few small spikes around epochs 330, 420 and 600, but the loss came back down each time, so the training was stable and did not blow up.

## 8. Results and Comparison

The results of all six models on the test set (6 samples) are in Table 13, sorted from best to worst by RMSE.

*Table 13: Comparative performance on the test set*

| Rank | Model | MAE | MSE | RMSE | R² |
| :---: | :--- | ---: | ---: | ---: | ---: |
| 1 | Linear Regression | 6,286.45 | 49,830,097 | 7,059.04 | 0.9024 |
| 2 | Random Forest | 6,563.44 | 59,801,030 | 7,733.11 | 0.8829 |
| 3 | Neural Network | 6,662.58 | 61,383,084 | 7,834.74 | 0.8798 |
| 4 | KNN | 6,905.99 | 65,979,671 | 8,122.79 | 0.8708 |
| 5 | XGBoost | 7,976.01 | 94,891,072 | 9,741.20 | 0.8142 |
| 6 | Decision Tree | 8,640.17 | 101,047,710 | 10,052.25 | 0.8022 |

Linear Regression is first in all four metrics. Its RMSE of 7,059 means the typical prediction error is about 7,000 in salary, and its R² of 0.9024 means it explains about 90% of the variation in the test salaries. Random Forest, Neural Network and KNN come next and are quite close to each other (RMSE between 7,733 and 8,123). XGBoost and the Decision Tree are the weakest, with RMSE above 9,700.

![Figure 3: Bar charts of MAE, MSE, RMSE and R² for all six models](images/cell31_6.png)

*Figure 3: Bar charts of MAE, MSE, RMSE and R² for all six models*

Figure 3 shows the same thing visually. Linear Regression has the shortest bars for MAE, MSE and RMSE and the tallest bar for R². Decision Tree and XGBoost have the tallest error bars.

### 8.1 Actual vs Predicted Values

![Figure 4: Actual vs predicted salary for each model on the test set](images/cell30_3.png)

*Figure 4: Actual vs predicted salary for each model on the test set*

In Figure 4 the dashed line is a perfect prediction, so the closer a point is to this line the better. The Linear Regression points are the closest to the line overall. The test sample with an actual salary of about 83,000 is predicted at about 75,000 by Linear Regression, but only about 66,000 to 70,000 by the other models, and the Decision Tree and XGBoost points are the furthest away. The sample with an actual salary near 114,000 is under-predicted by all models (about 101,000 to 104,000). The sample with actual salary near 112,600 is slightly over-predicted by all models.

![Figure 5: Actual and predicted salary for each of the 6 test samples](images/cell31_5.png)

*Figure 5: Actual and predicted salary for each of the 6 test samples*

Figure 5 gives the same comparison sample by sample (the labels on the x-axis are the years of experience of each test sample). For the sample with 3.3 years of experience all six models predict too low (about 54,000 to 58,000 against about 64,000 actual), so no model does well on that point. For 3.8 and 5.0 years the predictions are fairly close to the actual values.

![Figure 6: Fitted curves of all six models with the training and test points](images/cell31_4.png)

*Figure 6: Fitted curves of all six models with the training and test points*

Figure 6 shows how each model draws its prediction over the whole experience range. Linear Regression is a straight line that goes through the middle of the points. Decision Tree, Random Forest and XGBoost make steps, because they predict a constant value inside each region. The XGBoost steps are the biggest. KNN follows the training points closely and gets a bit wiggly, and the Neural Network is smooth in the middle with small bends near the start and the end. The black stars (test points) lie on both sides of the lines, which is why the errors are not small even for the best model.

### 8.2 Cross-Validation Check

The test set has only 6 rows, so one lucky or unlucky split can change the ranking. To check this I also ran 5-fold cross-validation on all 30 rows with the same settings. The results are in Table 14.

*Table 14: 5-fold cross-validation on the full dataset*

| Model | CV MAE | CV RMSE | CV RMSE std | CV R² |
| :--- | ---: | ---: | ---: | ---: |
| Random Forest | 4,647.04 | 5,327.41 | 1,472.36 | 0.9262 |
| Linear Regression | 4,726.55 | 5,694.82 | 1,159.60 | 0.9224 |
| KNN | 4,991.64 | 6,013.87 | 1,708.11 | 0.9062 |
| XGBoost | 5,449.71 | 6,490.49 | 1,980.05 | 0.8910 |
| Decision Tree | 5,683.22 | 6,822.62 | 2,115.72 | 0.8779 |
| Neural Network | 5,753.50 | 6,928.47 | 935.43 | 0.8985 |

In cross-validation Random Forest has the lowest average RMSE (5,327.41) and Linear Regression is second (5,694.82). The difference between them is about 370, which is smaller than the standard deviation across the folds (1,472 for Random Forest and 1,160 for Linear Regression). So these two models are practically equal here. The notebook itself also printed that the test-set winner (Linear Regression) and the cross-validation winner (Random Forest) are different and that the gap is within noise for such a small test set. Decision Tree, XGBoost and the Neural Network are at the bottom in cross-validation as well. The Neural Network has the smallest standard deviation (935), so it is the most consistent, but its average error is higher.

## 9. Strengths and Limitations of Each Algorithm

*Table 15: Strengths and limitations*

| Algorithm | Strengths | Limitations |
| :--- | :--- | :--- |
| Linear Regression | Very simple and fast. Only two parameters, easy to explain. Works very well when the data is linear like here. Can predict outside the training range. | Cannot learn curves. Affected by outliers. Assumes a straight-line relationship. |
| Decision Tree | Easy to understand and draw. No scaling needed. Can fit non-linear data. | Gives flat steps, not a smooth line. Overfits easily (the tuned tree here was fully grown). Cannot predict outside the training range. |
| Random Forest | Averaging many trees reduces overfitting and makes it more stable. Handles noise well. | Still gives step-like predictions. Slower and harder to explain than one tree. Too heavy for 1 feature and 24 rows. |
| Neural Network | Can learn complex non-linear patterns. Can use the GPU. Works with many features. | Needs scaling and many settings (layers, epochs, learning rate). Result changes with the random seed. 24 rows is too little data for it. Hard to interpret. |
| XGBoost | Very strong on large tabular data. Has regularisation. Fast on GPU. | Many hyper-parameters to tune. With so little data it only made coarse steps. Cannot predict outside the training range. |
| KNN | No training step. Simple idea. Follows local patterns in the data. | Depends on k and on scaling. Must keep all the data and is slow for big data. Becomes unreliable when points are few. Cannot predict outside the training range. |

## 10. Best Algorithm and Conclusion

Based on the experiment, the best algorithm is Linear Regression. On the common 80:20 test set it has the lowest MAE (6,286.45), the lowest MSE (49,830,097), the lowest RMSE (7,059.04) and the highest R² (0.9024) of all six models.

The reason is the shape of the data. The correlation between experience and salary is 0.9782 and the scatter plot is almost a straight line, so a straight-line model matches the data. It needs just two numbers (slope and intercept), so it can be fitted well even with 24 training rows. The other models are more flexible but they have to learn the same simple pattern from very few points. Decision Tree, Random Forest and XGBoost approximate the line with steps, and KNN averages neighbours, so they lose accuracy, especially at the ends of the range. The Neural Network has too many parameters for such a small dataset.

One point must be mentioned honestly. In the 5-fold cross-validation Random Forest was a little better than Linear Regression (RMSE 5,327 against 5,695), but this gap is smaller than the fold-to-fold variation, and the test set has only 6 samples. So Linear Regression and Random Forest are close and the ranking could change with another split. I still choose Linear Regression because it is the best on the required test split, it is nearly as good as Random Forest in cross-validation, and it is much simpler and easier to explain.

Overall, this lab showed me that the most complex algorithm is not always the best one. For a small dataset with one feature and a linear relationship, a simple model gave the best result. For bigger datasets with many features and non-linear patterns, Random Forest, XGBoost or a Neural Network would be a better choice.
