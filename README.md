Dimensionality Reduction Lab: PCA, LDA, and KPCA

Overview
This lab is based on Chapter 5 of Python Machine Learning (Second Edition) by Sebastian Raschka and Vahid Mirjalili. The chapter covers unsupervised and supervised dimensionality reduction methods, including Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Kernel Principal Component Analysis (KPCA).

In this hands-on lab, you will:
- Implement PCA from scratch and using scikit-learn.
- Apply LDA for supervised dimensionality reduction.
- Use KPCA to handle nonlinear data.
- Visualize results and analyze the impact of dimensionality reduction on classification tasks.

The lab uses the Wine dataset (from UCI) for PCA and LDA, and synthetic datasets (half-moons and circles) for KPCA.
Objectives
- Understand how PCA maximizes variance for unsupervised dimensionality reduction.
- Learn how LDA maximizes class separability using label information.
- Explore KPCA for nonlinear mappings using the RBF kernel.
- Compare the performance of these techniques on real and synthetic data.
- Visualize transformed data and evaluate using a simple classifier (e.g., Logistic Regression).
  
Prerequisites
Python 3.x
Libraries: numpy, pandas, matplotlib, scikit-learn, scipy
Install via pip: pip install numpy pandas matplotlib scikit-learn scipy
Jupyter Notebook (recommended for interactive execution)


Setup
1. Clone this repository:
   git clone <repo-url>
2. Navigate to the lab directory:
   cd lab-dimensionality-reduction
3. Launch Jupyter:
   jupyter notebook
4. Open dimensionality_reduction_lab.ipynb (or create one with the code below).
5. Download the Wine dataset if needed: wine.data
   
Part 1: Principal Component Analysis (PCA)
Unsupervised technique that finds principal components (directions of maximum variance) to reduce dimensionality.

Part 2: Linear Discriminant Analysis (LDA)
Supervised technique that maximizes class separability using label information.

Part 3: Kernel Principal Component Analysis (KPCA)
Nonlinear dimensionality reduction using the RBF kernel, applied to half-moon and circle datasets.


ANALYSIS QUESTIONS

1. Explained Variance (PCA)

As the number of principal components increases, the explained variance increases because each component captures additional variability in the dataset.
The first few components usually explain most of the variance, while later components contribute very little.
For the Wine dataset, typically around 6–7 components are enough to explain ~95% of the variance.

2. PCA vs. LDA (Wine dataset)

PCA is unsupervised: it projects data in directions of maximum variance without considering class labels.
LDA is supervised: it explicitly uses class labels to maximize the separation between classes while minimizing within-class variance.
As a result, LDA typically outperforms PCA for classification tasks, since PCA may retain variance that is irrelevant for class discrimination, while LDA focuses only on discriminative information.

3. KPCA Gamma Parameter (Half-Moon dataset)

Small γ (e.g., 0.01): The RBF kernel becomes very broad → data points appear more similar → transformation does not capture local structure → classes remain poorly separable.
Large γ (e.g., 100): The RBF kernel becomes too narrow → transformation is dominated by noise → overfitting occurs and decision boundaries become too fragmented.
Moderate γ (e.g., 10–20): Produces a good mapping → half-moon classes become approximately linearly separable in the transformed space.

4. Classifier Performance

Original data: Logistic Regression or SVM struggles if classes are not linearly separable (e.g., half-moons or circles).
PCA-transformed data: Classification may improve slightly, but since PCA ignores class labels, performance gain is limited.
LDA-transformed data: Performance improves significantly because features are explicitly optimized for class separability.

Computation time:

PCA and LDA preprocessing add some overhead but reduce feature dimensions, which speeds up classification on large datasets.
LDA often yields the best trade-off: higher accuracy and reduced computation.

Limitations:

When might standard PCA fail?
Standard PCA fails when the dataset is not linearly separable or when important information lies in nonlinear relationships between features.
Example: In the two concentric circles dataset, PCA cannot separate the classes since variance-based linear projections cannot unfold the circular structure.

How does KPCA address this?
Kernel PCA (KPCA) extends PCA by applying the kernel trick, which implicitly maps the data into a higher-dimensional feature space.
In this space, nonlinear structures (e.g., circles, half-moons) become linearly separable.
By using kernels such as the RBF kernel, KPCA can uncover complex nonlinear patterns that standard PCA cannot capture.


