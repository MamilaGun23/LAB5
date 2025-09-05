#  Dimensionality Reduction Lab: PCA, LDA, and KPCA

This lab explores three powerful dimensionality reduction techniques—Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Kernel PCA (KPCA)—and their impact on visualization and classification performance.

##  Installation
Install required dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

##  Setup
```bash
# Clone this repository
git clone <repo-url>

# Navigate to the lab directory
cd lab-dimensionality-reduction

# Launch Jupyter Notebook
jupyter notebook
```
Open `dimensionality_reduction_lab.ipynb` and run the cells.

Download the Wine dataset if needed: [`wine.data`](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)

##  Part 1: Principal Component Analysis (PCA)
**Type:** Unsupervised  
- Identifies directions of maximum variance to reduce dimensionality  
- Implement PCA from scratch or using `scikit-learn`  
- Visualize the first 2 principal components and plot class distributions  
- Evaluate explained variance (~95% typically requires 6–7 components for the Wine dataset)

##  Part 2: Linear Discriminant Analysis (LDA)
**Type:** Supervised  
- Maximizes class separability using label information  
- Reduces dimensionality while optimizing for discrimination  
- Visualize transformed data using 2D LDA components  
- Compare classifier performance on original vs LDA-transformed data

##  Part 3: Kernel Principal Component Analysis (KPCA)
**Type:** Nonlinear  
- Uses RBF kernel to handle nonlinear data structures  
- Apply KPCA on synthetic datasets (e.g., half-moons, circles)  
- Explore the effect of the gamma (γ) parameter:  
  - Small γ → underfitting, poor class separation  
  - Large γ → overfitting, fragmented decision boundaries  
  - Moderate γ → optimal class separation  
- Visualize KPCA-transformed data for different γ values

##  Analysis Questions

### PCA Explained Variance
- Explained variance increases with number of components  
- First few components capture most variance  
- Wine dataset: ~6–7 components → ~95% variance

### PCA vs LDA (Wine Dataset)
| Technique | Type        | Focus                          | Classification |
|-----------|-------------|--------------------------------|----------------|
| PCA       | Unsupervised| Maximizes variance             | Moderate       |
| LDA       | Supervised  | Maximizes class separation     | High           |

### KPCA Gamma Parameter (Half-Moon Dataset)
- Small γ → underfitting  
- Large γ → overfitting  
- Moderate γ → optimal separation

### Classifier Performance
- Logistic Regression / SVM struggle on nonlinear data  
- PCA may help slightly but ignores labels  
- LDA improves performance by optimizing for class separability

### Computation Time
- PCA & LDA reduce feature dimensions → faster classification  
- LDA offers high accuracy with reduced computation

##  Limitations
- PCA fails on nonlinear datasets or when key info lies in nonlinear relationships  
- KPCA handles nonlinear patterns using kernels (e.g., RBF)

##  Resources
- Raschka, S., & Mirjalili, V. (2017). *Python Machine Learning* (2nd Edition), Chapter 5  
- [Scikit-learn Documentation](https://scikit-learn.org)  
- [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)

##  About
This lab demonstrates:
- **Unsupervised** dimensionality reduction with PCA  
- **Supervised** reduction with LDA  
- **Nonlinear** reduction with KPCA  


