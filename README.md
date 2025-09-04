Kernel PCA and Dimensionality Reduction — Lab 5

This repository contains an exploration of dimensionality reduction techniques including PCA (Principal Component Analysis), LDA (Linear Discriminant Analysis), and Kernel PCA (RBF Kernel PCA).

We use synthetic datasets (e.g., two moons) and real datasets (e.g., Wine dataset) to compare the effectiveness of different approaches in data visualization and classification.

Contents:

LAB5.ipynb: Main Jupyter notebook containing all code and analysis.

Implementation of:

Standard PCA

LDA

RBF Kernel PCA (custom implementation using NumPy/SciPy)

Logistic Regression / SVM classifiers for evaluation

Visualizations of decision boundaries and feature transformations.

Installation & Setup:

Clone the repository and install dependencies:

git clone https://github.com/yourusername/LAB5.git
cd LAB5
pip install -r requirements.txt


Dependencies:

numpy

scipy

scikit-learn

matplotlib

Usage:

Open the Jupyter notebook:

jupyter notebook LAB5.ipynb


Run the cells step by step to:

Apply PCA, LDA, and KPCA.

Visualize decision regions and transformed features.

Train classifiers and compare performance.

Analysis Questions & Answers
1. Explained Variance in PCA

How it changes: The explained variance ratio decreases as the number of components increases. The first few components usually capture most of the variance.

95% threshold: For the Wine dataset, typically the first 2–3 components capture around 95% of the variance. The exact number depends on dataset scaling and preprocessing.

2. PCA vs. LDA on the Wine Dataset

PCA: Maximizes variance in the data without considering class labels. Good for compression/visualization but not necessarily optimal for classification.

LDA: Finds axes that maximize separation between classes. Since it uses label information, it often provides features that are more discriminative.

Why LDA performs better: In classification tasks, LDA aligns the projection with class boundaries, leading to higher separability and usually better classifier performance than PCA.

3. KPCA Gamma Parameter (γ)

Small γ (e.g., 0.01): The RBF kernel becomes too “wide.” All points look similar, and the kernel matrix approaches a constant matrix. The transformed features lose structure, making classes harder to separate.

Large γ (e.g., 100): The kernel becomes too “narrow.” Each point is only similar to itself, causing overfitting. The transformed space may scatter points too far apart and reduce generalization.

Balanced γ (e.g., 15): Preserves the nonlinear structure (like in the half-moon dataset), making classes linearly separable in the new feature space.

4. Classifier Performance (SVM / Logistic Regression)

Original Data: Works well only if data is linearly separable; otherwise accuracy drops.

PCA-transformed Data: Reduces dimensionality but may lose discriminative power since class information is not used.

LDA-transformed Data (Wine): Usually achieves the best accuracy because projections maximize class separation.

Observation: LDA often improves classification accuracy and reduces computation time compared to high-dimensional original data.

Limitations:

When PCA Fails:
PCA assumes linear relationships. It fails on nonlinear datasets like concentric circles or half-moons where variance alone does not capture class structure.

How KPCA Helps:
KPCA uses a kernel trick (e.g., RBF) to project data into a higher-dimensional feature space, where nonlinear structures become linearly separable. This allows handling of curved or complex decision boundaries that PCA cannot manage.

Conclusion:

PCA is useful for dimensionality reduction but not always optimal for classification.

LDA leverages label information, making it powerful for supervised tasks.

Kernel PCA extends PCA to nonlinear cases, making it effective for datasets like half-moons.

Choice of technique depends on dataset structure and task (compression vs. classification).
