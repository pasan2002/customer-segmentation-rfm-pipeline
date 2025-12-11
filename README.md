# Customer Segmentation Analysis - Online Retail II

## 📌 Project Overview
This project performs a comprehensive customer segmentation analysis using the **Online Retail II** dataset. The goal is to identify distinct customer groups based on purchasing behavior to enable targeted marketing strategies and improved customer relationship management (CRM).

The project implements and compares multiple unsupervised machine learning algorithms to determine the most effective segmentation approach.

## 📊 Dataset
The dataset used is the **Online Retail II** dataset from the UCI Machine Learning Repository.
- **Source**: [UCI Machine Learning Repository - Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Description**: This dataset contains all the transactions occurring for a UK-based and registered, non-store online retail between 01/12/2009 and 09/12/2011.

## 🛠 Technologies & Algorithms
The project utilizes Python and several machine learning libraries.

### Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `scipy`

### Clustering Algorithms Implemented
The following algorithms were implemented and compared:
1.  **K-Means Clustering**: A centroid-based algorithm.
2.  **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Good for arbitrary shapes and noise handling.
3.  **Mean Shift Clustering**: A centroid-based algorithm that works by updating candidates for centroids to be the mean of the points within a given region.
4.  **Agglomerative Hierarchical Clustering**: Builds a hierarchy of clusters.
5.  **Gaussian Mixture Models (GMM)**: A probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions.

## 📂 Project Structure

```
├── Data/                   # Dataset files (excluded from version control)
├── outputs/                # General output files
├── results/                # Model artifacts and visualization results
│   ├── eda_visualization/  # Exploratory Data Analysis plots
│   ├── models/             # Saved trained models (.joblib)
│   └── outputs/            # Clustered data and model cards
├── src/                    # Source code (Jupyter Notebooks)
│   ├── Aglomerative.ipynb
│   ├── DBScan.ipynb
│   ├── GMM.ipynb
│   ├── KMEANS.ipynb
│   ├── MEANSHIFT.ipynb
│   └── final_model_comparison.ipynb
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed along with the following packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

### Running the Analysis
1.  Clone the repository.
2.  Download the dataset from the [UCI link](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in the `Data/` directory.
3.  Navigate to the `src/` directory.
4.  Run the notebooks in the following order (recommended):
    - Individual algorithm notebooks (`KMEANS.ipynb`, `GMM.ipynb`, etc.) for detailed training and tuning.
    - `final_model_comparison.ipynb` to see the comparative analysis and final results.

## 📈 Results
The project evaluates models using metrics such as:
- **Silhouette Score**
- **Davies-Bouldin Score**
- **Calinski-Harabasz Score**

Based on the file structure, **Gaussian Mixture Models (GMM)** appeared to yield significant results, with optimized models saved in the `results/models/` directory.

## 📝 License
This project is open-source and available for educational and research purposes.
