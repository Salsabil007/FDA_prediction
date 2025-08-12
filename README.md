# FDA Prediction – Identifying Promising Biomedical Research Topics

## 📌 Overview
This project aims to **predict promising biomedical research topics that are likely to lead to future FDA-approved therapeutics**, years before the actual approval date.  
Unlike trial-level prediction models, our approach operates at the **scientific topic level**, leveraging large-scale biomedical literature to detect translational signals and guide early-stage resource allocation for drug development.  

By identifying high-potential research clusters early, this system can help accelerate the development of **novel therapeutics** and optimize decision-making in biomedical R&D.

---

## 🚀 Key Features
- **Scientific Literature Network Construction** – Built a direct citation network from **PubMed** data to capture relationships between biomedical research articles.  
- **Research Topic Clustering** – Used the **Leiden algorithm** to detect high-cohesion research clusters from citation networks.  
- **Drug Mapping to Clusters** – Linked FDA-approved drugs to clusters via pivotal trial publications.  
- **Predictive Feature Engineering** – Extracted features such as:  
  - Recent publication growth  
  - Human and animal study focus  
  - Basic vs. clinical research focus  
  - Citation impact within a cluster  
- **Machine Learning Model** – Trained a classifier to predict the likelihood of a cluster producing an FDA-approved drug in the future.  
- **Early Detection** – Achieved **4–6 years lead time** for >50% of exemplar drug clusters before FDA approval.  

---

## 📊 Results
| Dataset | Identified Before Approval | After Approval | False Negative |
|---------|----------------------------|---------------|----------------|
| Exemplar Drugs (n=99) | 81 | 4 | 14 |
| Out-of-Sample Blockbusters (n=10) | 8 | 2 | 0 |

**Performance Highlights:**  
- Sustained high prediction scores years before drug approval.  
- Strong separation in feature distributions between positive and negative clusters.  
- Accurate disease and drug association predictions from MeSH term rankings.  

---

## 🧠 Methods

### 1️⃣ Data Collection & Network Building
- Source: **PubMed** citation data.  
- Built **direct citation network** for biomedical literature.  

### 2️⃣ Topic Clustering
- Algorithm: **Leiden community detection** (modularity maximization).  

### 3️⃣ Drug Mapping
- Mapped clusters to drugs using pivotal trial publications.  

### 4️⃣ Feature Engineering
- Publication trends (growth rate, recency)  
- Study type distribution (human, animal, clinical, basic research)  
- Citation impact metrics  

### 5️⃣ Prediction Model
- Supervised ML classification using engineered features.  
- Output: **Prediction probability** of cluster producing an FDA-approved drug.  

---

## 📈 Early Detection Examples
- For exemplar drugs, high prediction scores (>0.75) appeared **4–6 years before FDA approval** for over half the cases.  
- Out-of-sample blockbusters were identified early in **80% of cases**.

---

## 🔮 Future Work
- Incorporate **trial-specific data** to improve accuracy.  
- Extend framework to predict **breakthrough patents**.  
- Explore integration with **deep learning models** for automated feature extraction.  

---


---

## ⚙️ Requirements
- Python 3.8+  
- pandas, numpy, scikit-learn  
- networkx, leidenalg  
- matplotlib, seaborn  


---

## 👩‍💻 Authors
- **Salsabil Arabi** – University of Wisconsin–Madison  
- **B. Ian Hutchins** – University of Wisconsin–Madison  

---

## 📜 References
1. Fu, T. et al. (2022). HINT: Hierarchical Interaction Network for Clinical-Trial-Outcome Predictions. *Patterns*, 3(7), 100445. [https://doi.org/10.1016/j.patter.2022.100445](https://doi.org/10.1016/j.patter.2022.100445)  
2. Traag, V. et al. (2019). From Louvain to Leiden: Guaranteeing Well-Connected Communities. *Scientific Reports*, 9, 5233. [https://doi.org/10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z)  

---
