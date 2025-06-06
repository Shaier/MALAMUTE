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
rm -rf data && unzip -o data.zip -d data && rm data.zip
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

### Citation

If you use this code or dataset, please cite us:

@misc{shaier2025malamutemultilingualhighlygranulartemplatefree,
  title     = {MALAMUTE: A Multilingual, Highly-granular, Template-free, Education-based Probing Dataset},
  author    = {Sagi Shaier and George Arthur Baker and Chiranthan Sridhar and Lawrence E Hunter and Katharina von der Wense},
  year      = {2025},
  eprint    = {2412.10105},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2412.10105}
}