import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

# import os
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        # Prefill
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    
    ### === TODO: Load your model (you may change this part) ===
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"   
    model_name = "shuyuej/Llama-3.2-3B-Instruct-GPTQ"
    spec_model_name = "shuyuej/Llama-3.2-1B-Instruct-GPTQ"

    # base_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     device_map='cuda:1',
    # )

    compilation_config = CompilationConfig(
        cudagraph_capture_sizes=[1],
    )
    num_spec_tokens = 3

    model = LLM(
        model_name,
        dtype=torch.float16,
        max_model_len=2048,
        quantization="gptq", #bitsandbytes
        tensor_parallel_size=1,
        #enforce_eager=True,
        speculative_config={
            "model": spec_model_name,
            "quantization": "gptq",
            "num_speculative_tokens": num_spec_tokens,
            "draft_tensor_parallel_size": 1,
            "max_model_len": 2048
        },
        compilation_config=compilation_config
    )
    #####################################
    
    # model.eval() 
    
    

    warmup_prompt = "Explain what AI is."
     
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
    )
    ####################################################################
    
    for i in tqdm(range(5), desc="Warm Up..."):
        _ = model.generate([warmup_prompt], sampling_params)
        
    prompt = "How to learn a new language?"

    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        generated = model.generate([prompt], sampling_params)

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = max_new_tokens / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)
        
    # response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    response = generated[0].outputs[0].text
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    # ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    # ppl = evaluate_ppl(model, tokenizer, device)
    # print(f"Perplexity (PPL): {ppl}")

    acceptance_counts = [0] * (num_spec_tokens + 1)
    for out in generated:
        for step, count in enumerate(out.metrics.spec_token_acceptance_counts):
            acceptance_counts[step] += count

    print(f"mean acceptance length: \
        {sum(acceptance_counts) / acceptance_counts[0]:.2f}")

    torch.distributed.destroy_process_group()
    # # Save results to CSV
    # import csv
    # rounded_tput = round(org_tput, 1)
    # # ppl = round(ppl, 2)
    # ppl = 0

    # with open("result.csv", mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Id", "value"])
    #     writer.writerow([0, ppl])
    #     writer.writerow([1, rounded_tput])
        
if __name__ == '__main__':
    main()
