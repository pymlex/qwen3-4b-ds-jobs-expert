# CPT for Data Science Jobs on Qwen3 4B

## Overview

`Qwen/Qwen3-4B-Base` is a 4B base model suitable for custom post-training. In this project, we apply continuous pretraining (CPT) on the `pymlex/datasciencejobs-tg` dataset to make the model more fluent on machine learning and data science job posts, vacancy patterns, salary and hiring phrasing, and domain vocabulary for further usage on platforms like HeadHunter or LinkedIn.

## Dataset

The dataset contains 1,784 Telegram job-post records. Each row includes:

- `id`
- `date`
- `views`
- `text`

The raw post text is used as plain language modeling data. Each sample is converted into the following format:

```text
Date: YYYY-MM-DD HH:MM:SS
<job post text><eos>
````

Short texts under 50 tokens were removed after loading because they are usually not full job descriptions, but channel information, recruiter contacts, or references to earlier posts.

The final split is `90/5/5`:

* Train: 1,546 samples
* Validation: 85 samples
* Test: 86 samples

The token distribution is shown below:

![Untitled](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/-S9aG4B8saeH0EJUeqU7V.png)

## CPT

Continued pretraining was performed with the following setup:

* GPU: NVIDIA GeForce RTX 5090
* VRAM: 31.35 GB
* CPU: Ryzen 9 9950X
* RAM: 64 GB

Training settings:

* max sequence length: `1024`
* batch size: `6`
* gradient accumulation: `4`
* epochs: `3`
* learning rate: `1e-4`
* scheduler: cosine
* optimiser: `adamw_torch`
* LoRA rank: `16`
* LoRA alpha: `32`
* LoRA dropout: `0.05`

The best checkpoint is selected by validation loss.

## Loss curves

The training and validation losses show a clear downward trend with normal oscillations for a small-domain CPT run.

![Untitled-1](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/yJBneVVFwKz1Yd2r43NFh.png)

## Inference

Use these two cells for inference.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "Qwen/Qwen3-4B-Base"
adapter_id = "pymlex/qwen3-4b-ds-jobs-expert"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
```

```python
def generate_continuation(model, tokenizer, prompt, max_new_tokens=260):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.08,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    decoded = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return decoded.strip()


sample_prompt = (
    "#вакансия #ml #python #удаленно\n"
    "Ищем Machine Learning Engineer в продуктовую команду. "
    "Задачи: построение и запуск моделей, работа с данными, эксперименты, "
    "поддержка ML-пайплайнов, участие в развитии продукта. "
)

output = generate_continuation(model, tokenizer, sample_prompt, max_new_tokens=1000)
print("Prompt:")
print(sample_prompt)
print("\nGenerated continuation:")
print(output)
```

## Perplexity evaluation

We evaluated both the base model and the tuned model on the test split.

| Model       | Perplexity |   Loss |
| ----------- | ---------: | -----: |
| Base Model  |      7.014 | 1.9479 |
| Tuned Model |      5.409 | 1.6881 |

The tuned model reduces perplexity by about `22.9%` and loss by about `13.3%`. This shows that CPT makes the model substantially more confident and accurate on domain-specific job-post text.
