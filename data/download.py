import os
import json
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
from datasets import load_dataset, load_from_disk
import datasets


def download_model():
    try:
        output_dir = snapshot_download(
            repo_id="FlagAlpha/Atom-7B",
            repo_type="model",
            local_dir="/workspace/data/Atom-7B",
            cache_dir="/workspace/cache",
            token="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",
            )
        print("success in", output_dir)
    except Exception as e:
        print('download fail:', e)

def download_dataset():
    try:
        output_dir = snapshot_download(
            repo_id="yahma/alpaca-cleaned",
            repo_type="dataset",
            local_dir="/workspace/data/alpaca-cleaned",
            cache_dir="/workspace/cache",
            token="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",
            )
        print("success in", output_dir)
    except Exception as e:
        print('download fail:', e)

def load_slim():
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_DATASETS_CACHE"] = "/mnt/local-nvme/SlimPajama-627B"
    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    ds = load_dataset("cerebras/SlimPajama-627B", cache_dir="/mnt/local-nvme/SlimPajama-627B")
    print(ds)
    # ds.save_to_disk("/mnt/local-nvme2/SlimPajama-627B")
    ds = load_from_disk("/mnt/local-nvme2/SlimPajama-627B")
    print(ds)

def download_tokenizers():
    try:
        output_dir = snapshot_download(
            repo_id="Xenova/llama-3-tokenizer",
            repo_type="model",
            local_dir="/workspace/data/llama-3-tokenizer",
            cache_dir="/workspace/cache",
            token="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",
            )
        print("success in", output_dir)
    except Exception as e:
        print('download fail:', e)


if __name__=='__main__':
    hf_token = "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"

    tokenizer_list = [
        "Xenova/Meta-Llama-3.1-Tokenizer",
        "Xenova/llama-3-tokenizer",
        "Xenova/llama2-tokenizer",
        "Xenova/llama2-chat-tokenizer",
        "Xenova/gpt-4o",
        "Xenova/gpt-3.5-turbo",
        "Xenova/gpt-4",
        "Xenova/gpt-3",
        "Xenova/gpt2",
    ]

    model_list = [
        "FlagAlpha/Atom-7B",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B",
    ]

    dataset_list = [
        "yahma/alpaca-cleaned",
        "ptb_text_only",
        "cerebras/SlimPajama-627B",
    ]