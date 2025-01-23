# <div align="center">LongRefiner | Hierarchical Document Refinement for Long-context Retrieval-augmented Generation</div>

<div align="center">
<p>
<a href="#️-installation">Installation</a> |
<a href="#-quick-start">Quick-Start</a> 
</p>
</div>


## 🔍 Overview

LongRefiner is an efficient plug-and-play refinement system for long-context RAG applications. It achieves 10x compression while maintaining superior performance through:

<div align="center">
<img src="/assets/main_figure.jpg" width="800px">
</div>

## ✨ Key Features

- ⚡ **Low Latency**: 10x computational cost reduction compared to baselines
- 🔌 **Plug and Play**: Compatible with any LLM and retrieval system
- 📑 **Hierarchical Document Structuring**: XML-based efficient document representation 
- 🔄 **Adaptive Refinement**: Dynamic content selection based on query requirements

## 🗺️ RoadMap

- [ ] Release training code
- [ ] Release trained modules


## 🛠️ Installation

```bash
git clone https://github.com/ignorejjj/LongRefiner
cd LongRefiner
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start

```python
import json
from longrefiner import LongRefiner

# Initialize
query_analysis_module_lora_path = "model/Qwen2.5-3B-Instruct-query-analysis"
doc_structuring_module_lora_path = "model/Qwen2.5-3B-Instruct-doc-structuring"
selection_module_lora_path = "model/Qwen2.5-3B-Instruct-global-selection"

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
longrefiner = LongRefiner()
refined_result = longrefiner.run(question, document_list, budget=2048)

print(refined_result)
```
