from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from peft import PeftModel, PeftConfig
import torch
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-3B-Instruct-lora")
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

gptq_config = GPTQConfig(bits=4, dataset='wikitext2', tokenizer=tokenizer)

quantized_model = AutoModelForCausalLM.from_pretrained("Llama-3.2-3B-Instruct-lora", device_map="auto", quantization_config=gptq_config)

# quantized_model = AutoModelForCausalLM.from_pretrained(
#     "facebook/opt-125m",
#     device_map="auto",
#     max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"},
#     quantization_config=gptq_config
# )

quantized_model.save_pretrained("Llama-3.2-3B-Instruct-gptq-lora")
tokenizer.save_pretrained("Llama-3.2-3B-Instruct-gptq-lora")

# # if quantized with device_map set
# quantized_model.to("cpu")
# quantized_model.save_pretrained("opt-125m-gptq")