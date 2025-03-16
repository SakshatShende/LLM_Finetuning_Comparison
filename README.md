#  LLM Finetuning Comparison 

**Just experimenting with LLM finetuning using various techniques and benchmarking scores for comparison.**  
This project focuses on **efficient LLM finetuning** using **parameter-efficient methods** and compares various **benchmark scores** across multiple approaches.

---

##  Model & Setup

###  Finetuning Base Model
- **LLama 3.2 3B Instruct** 

###  Model Loading
- **Implemented Flash Attention 2** for reduced training time 
- **Model loaded in 16-bit precision** (except in QLoRA, which uses **4-bit quantization** with **Double Quantization & NormalFloat4**) 

###  Dataset
- **Dataset Used**: ["gbharti/finance-alpaca"](https://huggingface.co/datasets/gbharti/finance-alpaca) 
(A financial domain instruction-tuning dataset)

---

##  Hyperparameters
| Parameter       | Value |
|----------------|-------|
| **Rank**       | 32    |
| **BF16**       | True  |
| **Max Steps**  | 200   |

- **For VB LoRA** → `num_vectors=60`, `vector_length=256`
- **For Ada LoRA** → `init_rank=32`, `target_r=4`

---

##  Hardware Used
- **GPU**: NVIDIA RTX **3090** (24GB VRAM) 
- **CPU**: **4 vCPUs**
- **RAM**: **31GB**

## Fine-Tuning Benchmark Results
Test Set - Questions not seen during training
Eval Set - Questions seen during training

| Method   | Peak Memory| Training Time| Adapter Size | F1 Test | F1 Eval | Cosine Sim Test| Cosine Sim Eval| BERT Score Test| BERT Score Eval| Avg Inference Time| Memory Used |
|----------|------------|--------------|--------------|---------|---------|----------------|----------------|----------------|----------------|------------------|--------------|
| Base     | ..         | ..           | ..           | 17.06   | 17.25   | 0.3600         | 0.4803         | 0.4100         | 0.4890         | 2.08 s           | 12.77 GB     |
| LoRA     | 20.49 GB   | 347.8 s      | 185.6 MB     | 20.12   | 19.798  | 0.4748         | 0.5540         | 0.5023         | 0.5330         | 3.89 s           | 12.95 GB     |
| QLoRA    | 14.2 GB    | 453.5 s      | 185.6 MB     | 17.25   | 17.95   | 0.4070         | 0.5027         | 0.4897         | 0.5012         | 3.92 s           | 12.19 GB     |
| AdaLoRA  | 20.68 GB   | 470.5 s      | 185.6 MB     | 17.75   | 18.53   | 0.4208         | 0.4620         | 0.4191         | 0.4451         | 4.8 s            | 12.95 GB     |
| VBLoRA   | 20.08 GB   | 427.3 s      | 1.3 MB       | 23.73   | 18.42   | 0.4400         | 0.4966         | 0.5098         | 0.5076         | 5.6 s            | 12.81 GB     |


<p align="center">
    <img src="https://github.com/user-attachments/assets/3df82db8-ee22-4c08-941c-a2d10482c8e3" width="30%" alt="Graph">
    <img src="https://github.com/user-attachments/assets/711a6f0e-4217-46e1-9106-63bcb9a4e1fc" width="30%" alt="Graph 2">
    <img src="https://github.com/user-attachments/assets/e5796c83-1b1d-466a-a174-3d66478a74c2" width="30%" alt="Graph 3">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/6232ad38-3222-4ced-aaf8-f09612edcfef" width="30%" alt="Graph 5">
    <img src="https://github.com/user-attachments/assets/a20ee4fa-edd4-4139-88b6-2e2f4fa1c8c1" width="30%" alt="Graph 6">
    <img src="https://github.com/user-attachments/assets/76a56d8f-6f7e-48f9-abab-7ab6e26e7552" width="30%" alt="Graph 4">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/c288b67c-0bbe-42ac-ade6-b19be6b2f6da" width="30%" alt="Graph 5">
    <img src="https://github.com/user-attachments/assets/26e56927-95e1-4d5b-868f-441d6668a074" width="30%" alt="Graph 6">
    <img src="https://github.com/user-attachments/assets/751d3851-e14f-4430-84c4-6c7942feecad" width="30%" alt="Graph 4">
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/b134292b-5aee-46f3-9f70-a067382ffe44" width="40%" alt="Graph 5">
    <img src="https://github.com/user-attachments/assets/e01a9b5c-bbfe-473c-86c5-c2fdf4ebb1aa" width="40%" alt="Graph 6">
    
</p>

As we can see, VBLora performs surprisingly well for a huge memory reduction in Adapter Size, although the inference time increases a bit. I don't know why AdaLora didnt perform well, maybe tuning the hyperparameters will result in a better score.
Next we will perform DPO,PPO on the base model to make our response more human-like.
