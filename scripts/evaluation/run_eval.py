import os
import json
import argparse
from longrefiner import LongRefiner
from flashrag.config import Config
from flashrag.evaluator import Evaluator
from flashrag.utils import get_generator, get_dataset
from flashrag.prompt_template import PromptTemplate


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run evaluation script for question answering")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument(
        "--generator_model", type=str, default="llama3.1-8B-instruct", help="Name of the generator model"
    )
    parser.add_argument(
        "--generator_model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path to the base model"
    )
    parser.add_argument(
        "--base_refiner_model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Path to the base model"
    )
    parser.add_argument(
        "--query_analysis_module_lora_path",
        type=str,
        default="model/Qwen2.5-3B-Instruct-query-analysis",
        help="Path to the query analysis module lora",
    )
    parser.add_argument(
        "--doc_structuring_module_lora_path",
        type=str,
        default="model/Qwen2.5-3B-Instruct-doc-structuring",
        help="Path to the doc structuring module lora",
    )
    parser.add_argument(
        "--global_selection_module_lora_path",
        type=str,
        default="model/Qwen2.5-3B-Instruct-global-selection",
        help="Path to the global selection module lora",
    )
    parser.add_argument("--score_model_name", type=str, default="bge-reranker-v2-m3", help="Name of the score model")
    parser.add_argument(
        "--score_model_path", type=str, default="BAAI/bge-reranker-v2-m3", help="Path to the score model"
    )
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3,4,5,6,7", help="GPU IDs to use")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--retrieval_result_path", type=str, default="sample_docs.json", help="Path to the all docs file"
    )
    parser.add_argument("--framework", type=str, default="vllm", help="Framework to use")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU memory utilization ratio")
    parser.add_argument("--generator_max_input_len", type=int, default=15000, help="Maximum input length for generator")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--test_sample_num", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--save_note", type=str, default="", help="Note to save with results")
    return parser.parse_args()


def get_prompt_template(config, dataset_name):
    """Get prompt template for the specified dataset"""
    if dataset_name not in ["eli5", "asqa"]:
        system_prompt = (
            "Find the useful content from the provided documents, then answer the question. "
            "Answer the question directly. Your response should be very concise. Please provide use 'So the final answer is:' as a prefix for the final answer."
            "\nOutput format:\nQuestion: What is the capital of France?\nResponse:The capital city of France is Paris.So the final answer is: Paris.\n\nThe following are given documents.\n\n{reference}"
        )
        user_prompt = "Answer the question directly. Your response should be very concise. Please provide use 'So the final answer is:' as a prefix for the final answer.\nQuestion: {question}\nResponse: "
    else:
        system_prompt = (
            "Find the useful content from the provided documents, then answer the question. "
            "Answer the question directly. Your response should be very detailed."
            "\n\nThe following are given documents.\n\n{reference}"
        )
        user_prompt = (
            "Answer the question directly. Your response should be very detailed.\nQuestion: {question}\nResponse: "
        )

    return PromptTemplate(config, system_prompt, user_prompt)


def run(args):
    """Run evaluation pipeline"""
    # Initialize configuration
    config_dict = {
        "generator_model": args.generator_model,
        "generator_model_path": args.generator_model_path,
        "gpu_id": args.gpu_id,
        "save_dir": args.save_dir,
        "framework": args.framework,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "generator_max_input_len": args.generator_max_input_len,
        "generation_params": {"max_tokens": args.max_tokens},
        "dataset_name": args.dataset_name,
        "test_sample_num": args.test_sample_num,
        "save_note": args.save_note,
    }

    # Initialize components
    config = Config(config_dict)
    generator = get_generator(config)
    refiner = LongRefiner(
        base_model_path=args.base_refiner_model_path,
        query_analysis_module_lora_path=args.query_analysis_module_lora_path,
        doc_structuring_module_lora_path=args.doc_structuring_module_lora_path,
        global_selection_module_lora_path=args.global_selection_module_lora_path,
        score_model_name=args.score_model_name,
        score_model_path=args.score_model_path,
        max_model_len=25000,
    )

    # Prepare data
    all_split = get_dataset(config)
    data = all_split[args.split]
    with open(args.retrieval_result_path, "r") as f:
        retrieval_result = json.load(f)

    # Get prompt template
    template = get_prompt_template(config, args.dataset_name)

    # Process data and generate answers
    questions = data.question
    retrieval_docs = [retrieval_result.get(question, []) for question in questions]
    refined_result = refiner.batch_run(questions, retrieval_docs, budget=2048)
    input_prompts = [
        template.get_string(question, retrieval_result=docs) for question, docs in zip(questions, refined_result)
    ]

    # Generate answers and evaluate
    print("Starting answer generation...")
    output_list = generator.generate(input_prompts)
    data.update_output("prompt", input_prompts)
    data.update_output("retrieval_results", retrieval_docs)
    data.update_output("refined_results", refined_result)
    data.update_output("raw_pred", output_list)
    new_output_list = [i.split("So the final answer is:")[-1].strip() for i in output_list]
    data.update_output("pred", new_output_list)

    evaluator = Evaluator(config)
    result = evaluator.evaluate(data)
    print(result)
    print("------------------------\n")


if __name__ == "__main__":
    args = parse_args()
    run(args)
