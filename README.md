# k-NLPmeans and k-LLMmeans
**Summaries as Centroids for Scalable and Interpretable Text Clustering**

This repository contains the official implementation of **k-NLPmeans** and **k-LLMmeans**, text clustering algorithms that leverage textual summaries for dynamic centroid generation. 

*We introduce k-NLPmeans and k-LLMmeans, text-clustering variants of k-means that periodically replace numeric centroids with textual summaries. The key idea—summary-as-centroid—retains k-means assignments in embedding space while producing human-readable, auditable cluster prototypes. The method is LLM-optional: k-NLPmeans uses lightweight, deterministic summarizers, enabling offline, low-cost, and stable operation; k-LLMmeans is a drop-in upgrade that uses an LLM for summaries under a fixed per-iteration budget whose cost does not grow with dataset size. We also present a mini-batch extension for real-time clustering of streaming text. Across diverse datasets, embedding models, and summarization strategies, our approach consistently outperforms classical baselines and matches the accuracy of recent LLM-based clustering—without extensive LLM calls. Finally, we provide a case study on sequential text streams and release a StackExchange-derived benchmark for evaluating streaming text clustering.*

## 📂 Repository Structure
- **`kLLMmeans.py`** – Core implementation containing the `kNLPmeans`, `kLLMmeans`, `miniBatchNLPMeans` and `miniBatchKLLMeans` functions. Change value for variable `OPENAI_KEY`, `LLAMA_KEY`, `CLAUDE_KEY`, `DEEPSEEK_KEY` with your own keys before running.
- **`data_loaders/`** – Contains scripts for loading and preprocessing all data.
- **`data_loaders/clean_stackexchange.csv`** – Contains the clean stackexchange dataset used in the paper (unzip first)
- **`processed_data/`** – Folder where processed datasets will be stored (must be generated first).
- **`results/`** – Folder where results will be stored. It contains notebooks to calculate average ACC, NMI, dist.
- **Notebooks:**
  - `offline_experiments.ipynb` – Reproduces offline experiments from the paper.
  - `sequential_experiments.ipynb` – Runs sequential experiments.
  - `case_study_AI.ipynb` – Conducts the AI-related case study.

## ⚡ Getting Started
1. **Preprocess Data**:  
   Before running the experiments, preprocess the datasets by executing:  
   - `data_loaders/preprocess_offline_data.ipynb`
   - `data_loaders/preprocess_stackexchange.ipynb`  

   This will generate the necessary files in the `processed_data/` folder.  
   _(If you'd like to skip this step, contact me, and I can provide it for you —see the paper for details.)_

2. **Run Experiments**:  
   Open the provided Jupyter notebooks to reproduce the results from the paper.
   
3. **Process results**:  
   Open the provided Jupyter notebooks process results and get average ACC, NMI, dist.
   
## 📜 Citation
If you use this code or any of the data provided in this repository in your research, please cite the official paper:  
[arXiv:2502.09667](https://arxiv.org/abs/2502.09667)

For the Stack Exchange dataset, please cite both:
- Our processed version: [arXiv:2502.09667](https://arxiv.org/abs/2502.09667)
- Original source: [Internet Archive](https://archive.org/download/stackexchange)

  _(If you'd like the raw dataset described in the paper, contact me, and I can provide it for you —see the paper for details.)_
---

For any inquiries, refer to the contact details in the paper.



