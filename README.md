# 🧠 Comparative Study of Single-Task and Multi-Task Learning for Medical Diagnosis

**A comparative evaluation of independent Single-Task Shallow Neural Networks and shared-representation Multi-Task Learning (MTL) across three medical datasets: HAM10000, PTB-XL, and CheXpert.**




\


---

## 📌 Overview

This project presents a comparative study of two neural network learning strategies for medical prediction tasks: **independent Single-Task Learning (STL)** and **shared-representation Multi-Task Learning (MTL)**.

The study investigates whether learning a common representation across multiple medical datasets can improve prediction performance compared with training separate models for individual tasks.

Three publicly available medical datasets are considered:

- **HAM10000:** Dermatoscopic skin lesion images representing seven diagnostic categories.
- **PTB-XL:** Electrocardiogram (ECG) data used for five prediction labels.
- **CheXpert:** Chest X-ray-related data used for five selected disease labels.

The Single-Task approach trains an independent shallow neural network for each dataset. The MTL approach uses dataset-specific input projection layers, a shared neural network layer, and separate task-specific output heads.

Both approaches are evaluated using Accuracy, Precision, Recall, F1-score, and Area Under the ROC Curve (AUC).

The experimental findings demonstrate that Multi-Task Learning does not consistently outperform independent Single-Task Learning across all datasets and evaluation metrics.

---

## 🎯 Research Objectives

- Compare independent Single-Task Learning with shared-representation Multi-Task Learning.
- Examine the effect of shared representations across different medical data modalities.
- Implement shallow neural network architectures for three medical prediction tasks.
- Evaluate model performance using multiple classification metrics.
- Analyse the influence of dataset characteristics and class imbalance.
- Identify the limitations of shared learning and differences in evaluation procedures.

---

## 📊 Datasets Used

| Dataset  | Medical Data Type                | Input Features | Prediction Outputs |
| -------- | -------------------------------- | -------------: | -----------------: |
| HAM10000 | Dermatoscopic skin lesion images |          3,072 |          7 classes |
| PTB-XL   | ECG data                         |             60 |           5 labels |
| CheXpert | Chest X-ray-related features     |             22 |           5 labels |

### Dataset Details

**1. HAM10000**

- Contains dermatoscopic images belonging to seven diagnostic categories.
- The project uses a 3,072-dimensional flattened input representation.
- Classification is performed across seven skin lesion categories: AKIEC, BCC, BKL, DF, MEL, NV, and VASC.

**2. PTB-XL**

- Contains electrocardiogram recordings.
- The project uses 60 input features.
- Five prediction labels are considered for the classification task.

**3. CheXpert**

- Contains chest X-ray-related medical observations.
- The project uses 22 input features and five selected disease labels.
- A masking mechanism is applied to handle missing or unconfirmed labels during loss calculation.

---

## 🏗️ Model Architecture

The project implements two learning approaches.

### 1. Single-Task Learning (STL)

In the Single-Task approach, three independent shallow neural networks are developed.

Each network is trained separately using its corresponding dataset. The learned parameters are specific to that prediction task, and representations are not shared across datasets.

```text
             HAM10000
                 |
                 v
       Independent Neural Network
                 |
                 v
          7 Class Outputs


               PTB-XL
                 |
                 v
       Independent Neural Network
                 |
                 v
          5 Prediction Outputs


              CheXpert
                 |
                 v
       Independent Neural Network
                 |
                 v
          5 Prediction Outputs
```

### 2. Multi-Task Learning (MTL)

The shared-representation architecture contains three dataset-specific input projection layers connected to a common shared layer.

**Architecture Configuration**

| Component                 | Configuration           |
| ------------------------- | ----------------------- |
| HAM10000 Input Projection | Linear(3072, 128)       |
| PTB-XL Input Projection   | Linear(60, 128)         |
| CheXpert Input Projection | Linear(22, 128)         |
| Shared Layer              | Linear(128, 128) + ReLU |
| HAM10000 Output Head      | 7 outputs               |
| PTB-XL Output Head        | 5 outputs               |
| CheXpert Output Head      | 5 outputs               |

```text
 HAM10000 Input       PTB-XL Input       CheXpert Input
   3072 Features       60 Features        22 Features
        |                   |                  |
        v                   v                  v
  Input Projection    Input Projection   Input Projection
    Linear(3072,128)   Linear(60,128)     Linear(22,128)
        |                   |                  |
        +-------------------+------------------+
                            |
                            v
                   SHARED LAYER
                   Linear(128,128)
                        ReLU
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
         HAM10000         PTB-XL        CheXpert
          Head             Head           Head
        7 Outputs        5 Outputs      5 Outputs
```

The shared layer learns a common representation, while the separate output heads preserve dataset-specific prediction tasks.

---

## ⚙️ Loss Functions

Different loss functions are used according to the prediction requirements of each dataset.

| Dataset  | Loss Function            | Additional Handling                              |
| -------- | ------------------------ | ------------------------------------------------ |
| HAM10000 | CrossEntropyLoss         | Original MTL configuration uses no class weights |
| PTB-XL   | BCEWithLogitsLoss        | Five prediction labels                           |
| CheXpert | Masked BCEWithLogitsLoss | Missing or unconfirmed labels are masked         |

The original HAM10000 MTL experiment uses an unweighted CrossEntropyLoss, which is an important consideration when interpreting its performance on minority classes.

---

## 🛠️ Technology Stack

| Technology   | Purpose                                        |
| ------------ | ---------------------------------------------- |
| Python       | Core programming language                      |
| PyTorch      | Neural network implementation and training     |
| NumPy        | Numerical operations and data preparation      |
| Pandas       | Dataset handling and analysis                  |
| Scikit-learn | Evaluation metrics and model assessment        |
| Matplotlib   | Training graphs and performance visualisations |
| Seaborn      | Confusion matrix visualisation                 |
| Google Colab | Development and execution environment          |

---

## 📈 Experimental Results

The two approaches were evaluated using five metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Area Under the ROC Curve (AUC)

### Single-Task Learning Results

| Dataset  | Accuracy | Precision | Recall | F1-score |    AUC |
| -------- | -------: | --------: | -----: | -------: | -----: |
| HAM10000 |   0.4945 |    0.2499 | 0.2925 |   0.2284 | 0.7533 |
| PTB-XL   |   0.3318 |    0.4739 | 0.7033 |   0.5613 | 0.7990 |
| CheXpert |   0.4120 |    0.3057 | 0.8077 |   0.3711 | 0.5813 |

### Multi-Task Learning Results

| Dataset  | Accuracy | Precision | Recall | F1-score |    AUC |
| -------- | -------: | --------: | -----: | -------: | -----: |
| HAM10000 |   0.6832 |    0.2871 | 0.1530 |   0.1341 | 0.7645 |
| PTB-XL   |   0.3681 |    0.6374 | 0.3992 |   0.4755 | 0.7726 |
| CheXpert |   0.4915 |    0.3194 | 0.9113 |   0.4730 | 0.7010 |

**Evaluation Note:** HAM10000 and PTB-XL comparisons use their recorded test-set results. CheXpert results use validation data, and the metric calculation methods differ between the Single-Task and MTL implementations. Therefore, the CheXpert values should be interpreted cautiously and should not be treated as conclusive evidence of MTL superiority.

---

## 🔍 Key Findings

### HAM10000

- MTL achieved higher Accuracy and AUC.
- Single-Task Learning achieved higher Recall and F1-score.
- The MTL confusion matrix showed a strong tendency to predict the majority NV class.
- Class imbalance and the use of an unweighted loss are important limitations.

### PTB-XL

- MTL achieved higher Accuracy and Precision.
- Single-Task Learning achieved higher Recall, F1-score and AUC.
- Shared representations improved some reported metrics but not all.

### CheXpert

- MTL showed higher reported values across all five metrics.
- Differences in metric calculation require cautious interpretation.
- Further evaluation using consistent procedures is necessary for a fair comparison.

### Overall Observation

The experiments indicate that shared-representation learning does not guarantee better performance for every medical prediction task. Model performance depends on dataset characteristics, class distributions, prediction objectives and evaluation procedures.

---

## 📊 Visualisations and Analysis

The project includes experimental visualisations such as:

- Single-Task training and validation loss curves.
- Single-Task training and validation accuracy curves.
- Multi-Task training and validation loss curves.
- HAM10000 MTL confusion matrix.
- Comparative performance bar charts.
- Dataset-wise performance difference graphs.
- Individual task training and validation loss plots.

These visualisations help analyse training behaviour and compare the two learning strategies.

---

## 📁 Project Structure

```text
Medical-Diagnosis-Comparative-Study-STL-vs-MTL/
│
├── notebooks/
│   └── Medical_MultiTask_Learning_Project.ipynb
│
├── results/
│   ├── training_curves/
│   ├── confusion_matrices/
│   └── comparison_graphs/
│
├── report/
│   └── Project_Report.pdf
│
├── README.md
└── requirements.txt
```

*This is a suggested repository organisation. Add only the folders and files that are actually included in your GitHub repository.*

---

## 🚀 Running the Project

The experiments were developed and executed using Google Colab.

### Prerequisites

- Python 3.x
- Google account
- Google Colab
- Required Python libraries
- Access to the datasets used in the experiments

### Step 1: Clone the Repository

```bash
git clone https://github.com/thaslimaan6-tech/Medical-Diagnosis-Comparative-Study-STL-vs-MTL.git
```

### Step 2: Open the Notebook

Open the project notebook:

`notebooks/Medical_MultiTask_Learning_Project.ipynb`

Alternatively, upload the notebook directly to Google Colab.

### Step 3: Install Dependencies

Install the libraries required by the notebook. For example:

```python
!pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

Check the notebook for any additional dependencies before execution.

### Step 4: Configure the Datasets

Download the HAM10000, PTB-XL and CheXpert datasets from their respective official sources and configure the dataset paths according to the notebook.

Dataset access and any applicable terms of use must be followed.

### Step 5: Execute the Notebook

Run the notebook cells in sequence to perform data preparation, model implementation, training, evaluation and visualisation.

**Note:** Full execution may require substantial storage and computational resources, depending on dataset availability and hardware. The repository should not claim that the experiments can be reproduced instantly on every system.

---

## ⚠️ Limitations

- HAM10000 contains an imbalanced distribution across its seven classes.
- The original HAM10000 MTL model does not use class-weighted loss.
- Shared representations may introduce negative transfer between tasks.
- CheXpert metric calculations differ between the two implementations.
- The experiments use shallow neural network architectures.
- The reported results do not establish clinical validity or readiness for medical deployment.

---

## 🚀 Future Enhancements

- Investigate class-weighted loss functions for imbalanced medical datasets.
- Apply consistent evaluation procedures across all datasets.
- Explore alternative shared-representation architectures.
- Investigate negative transfer between medical prediction tasks.
- Evaluate models using independent and consistently defined test sets.
- Explore deeper neural network architectures and explainability methods.

---

## 🎓 Learning Outcomes

This project provided practical experience in:

- Implementing Single-Task and Multi-Task Learning.
- Developing shallow neural networks using PyTorch.
- Working with different medical data modalities.
- Designing shared-representation architectures.
- Applying dataset-specific loss functions.
- Evaluating models using multiple classification metrics.
- Analysing class imbalance and negative transfer.
- Creating training visualisations and comparative performance graphs.
- Conducting experimental research using Google Colab.

---

## 👩‍💻 Project Contributors

**Shaik Aslimaan Thaslimaan**

---

## 📚 References

1. Tschandl, P., Rosendahl, C., and Kittler, H. (2018). *The HAM10000 Dataset, a Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions.* Scientific Data, 5, 180161. [https://doi.org/10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

2. Wagner, P. et al. (2020). *PTB-XL, a Large Publicly Available Electrocardiography Dataset.* Scientific Data, 7, 154. [https://doi.org/10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

3. Irvin, J. et al. (2019). *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.* AAAI Conference on Artificial Intelligence, 33(1), 590–597. [https://doi.org/10.1609/aaai.v33i01.3301590](https://doi.org/10.1609/aaai.v33i01.3301590)

4. Caruana, R. (1997). *Multitask Learning.* Machine Learning, 28, 41–75. [https://doi.org/10.1023/A:1007379606734](https://doi.org/10.1023/A:1007379606734)

5. Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems, 32.

---

## 📄 License

This project was developed for academic and educational research purposes.

