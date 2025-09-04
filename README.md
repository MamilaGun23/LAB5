
**Kernel PCA and Dimensionality Reduction – Lab 5**

This document explores dimensionality reduction techniques including PCA, LDA, and Kernel PCA using synthetic and real datasets like the two moons and Wine dataset.

---

 **Contents**
- **LAB5.ipynb**: Main Jupyter notebook with code and analysis  
- **Implemented Techniques**:  
  - Standard PCA  
  - LDA  
  - RBF Kernel PCA (custom NumPy/SciPy implementation)  
  - Logistic Regression / SVM classifiers  
- **Visualizations**: Decision boundaries and feature transformations

---

 **Installation & Setup**
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/LAB5.git
cd LAB5
pip install -r requirements.txt
```

**Dependencies**:
- numpy  
- scipy  
- scikit-learn  
- matplotlib

---

 **Usage**
Open the Jupyter notebook:
```bash
jupyter notebook LAB5.ipynb
```

Run the cells to:
- Apply PCA, LDA, and KPCA  
- Visualize decision regions and transformed features  
- Train classifiers and compare performance

---

 **Analysis Questions & Answers**

**1. Explained Variance in PCA**  
- Explained variance ratio decreases with more components  
- First 2–3 components often capture ~95% of variance (Wine dataset)

**2. PCA vs. LDA on the Wine Dataset**  
- PCA: Maximizes variance, ignores class labels  
- LDA: Maximizes class separation using label info  
- LDA often yields better classification performance

**3. KPCA Gamma Parameter (γ)**  
- Small γ (e.g., 0.01): Kernel too wide, poor separation  
- Large γ (e.g., 100): Kernel too narrow, overfitting  
- Balanced γ (e.g., 15): Preserves nonlinear structure

**4. Classifier Performance**  
- Original Data: Works if linearly separable  
- PCA: May lose class info  
- LDA: Best accuracy due to class separation  
- LDA improves accuracy and reduces computation

---

 **Limitations**
- **PCA Fails**: On nonlinear datasets (e.g., concentric circles)  
- **KPCA Helps**: Projects data into higher-dimensional space for better separation

---

 **Conclusion**
- PCA: Good for compression, not always for classification  
- LDA: Powerful for supervised tasks  
- KPCA: Effective for nonlinear datasets  
- Technique choice depends on task and data structure

