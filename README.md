# MALAMUTE
**Multilingual, Highly-granular, Template-free, Education-based Probing Dataset**
MALAMUTE is a benchmark designed to evaluate language models on factual knowledge across diverse languages and educational levels. It avoids templates and emphasizes nuanced, real-world understanding.

---
## 🚀 Getting Started
### Prerequisites
- Python 3.11
- [Conda](https://docs.conda.io/en/latest/) (for environment management)
---
### Installation
#### 1. Clone the Repository
```bash
git clone https://github.com/Shaier/MALAMUTE.git
cd MALAMUTE
```
#### 2. Create a Conda Environment
```bash
conda create -n malamute python=3.11
conda activate malamute
```
#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
---
## 📂 Prepare the Data
Unzip the dataset and remove any extraneous files:
```bash
unzip -o data.zip -d data && rm data.zip && [ -d data/__MACOSX ] && rm -rf data/__MACOSX
```
---
## 🧪 Running Evaluations
### Masked Language Models (MLMs)
To evaluate using MLMs (e.g., BERT-style models):
```bash
python test_MLM.py
```
### Causal Language Models (CLMs)
To evaluate using CLMs (e.g., GPT-style models):
```bash
python test_CLM.py
```
---