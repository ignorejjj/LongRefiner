# <div align="center">LongRefiner | Hierarchical Document Refinement for Long-context Retrieval-augmented Generation</div>

<div align="center">
<p>
<a href="#️-installation">Installation</a> |
<a href="#-quick-start">Quick-Start</a> |
<a href="#-training">Training</a> |
<a href="#-evaluation">Evaluation</a> |
<a href='https://huggingface.co/collections/jinjiajie/longrefiner-683ac32af1dc861d4c5d00e2'>Huggingface Models</a>
</p>
</div>

## 🔍 Overview

LongRefiner is an efficient plug-and-play refinement system for long-context RAG applications. It achieves 10x compression while maintaining superior performance through hierarchical document refinement.

<div align="center">
<img src="/assets/main_figure.jpg" width="800px">
</div>

## ✨ Key Features

- ⚡ **Low Latency**: 10x computational cost reduction compared to baselines
- 🔌 **Plug and Play**: Compatible with any LLM and retrieval system
- 📑 **Hierarchical Document Structuring**: XML-based efficient document representation 
- 🔄 **Adaptive Refinement**: Dynamic content selection based on query requirements

## 🗺️ RoadMap

- [x] Release training code
- [x] Release trained modules
- [x] Release evaluation code
- [ ] Release code for building custom training data

## 🛠️ Installation

```bash
cd LongRefiner
pip install -r requirements.txt
pip install -e .
```

For training purposes, please additionally install the `Llama-Factory` framework by following the instructions in the [official repository](https://github.com/hiyouga/LLaMA-Factory):

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## 🚀 Quick Start

You can download the pre-trained LoRA models from [here](https://huggingface.co/collections/jinjiajie/longrefiner-683ac32af1dc861d4c5d00e2).

```python
import json
from longrefiner import LongRefiner

# Initialize
query_analysis_module_lora_path = "jinjiajie/Query-Analysis-Qwen2.5-3B-Instruct"
doc_structuring_module_lora_path = "jinjiajie/Doc-Structuring-Qwen2.5-3B-Instruct"
selection_module_lora_path = "jinjiajie/Global-Selection-Qwen2.5-3B-Instruct"

refiner = LongRefiner(
    base_model_path="Qwen/Qwen2.5-3B-Instruct",
    query_analysis_module_lora_path=query_analysis_module_lora_path,
    doc_structuring_module_lora_path=doc_structuring_module_lora_path,
    global_selection_module_lora_path=selection_module_lora_path,
    score_model_name="bge-reranker-v2-m3",
    score_model_path="BAAI/bge-reranker-v2-m3",
    max_model_len=25000,
)

# Load sample data
with open("assets/sample_data.json", "r") as f:
    data = json.load(f)
question = list(data.keys())[0]
document_list = list(data.values())[0]

# Process documents
refined_result = refiner.run(question, document_list, budget=2048)
print(refined_result)
```

## 📚 Training

Before training, prepare the datasets for three tasks in JSON format. Reference samples can be found in the training_data folder. We use the `Llama-Factory` framework for training. After setting up the training data, run:

```bash
cd scripts/training
# Train query analysis module
llamafactory-cli train train_config_step1.yaml  
# Train doc structuring module
llamafactory-cli train train_config_step2.yaml  
# Train global selection module
llamafactory-cli train train_config_step3.yaml  
```

## 📊 Evaluation

We use the [FlashRAG framework](https://github.com/RUC-NLPIR/FlashRAG) for RAG task evaluation. Required files:

- Evaluation dataset (recommended to obtain from FlashRAG's [official repository](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets))
- Retrieval results for each query in the dataset
- Model paths (same as above)

After preparation, configure the paths in `scripts/evaluation/run_eval.sh` and run:

```bash
cd scripts/evaluation
bash run_eval.sh
```


