# SAFE-QAQ: End-to-End Slow-Thinking Audio-Text Fraud Detection via Reinforcement Learning

SAFE-QAQ is an end-to-end framework for audio-text fraud detection that leverages reinforcement learning to enable slow-thinking decision-making. Below are instructions for setting up the environment, training the model, and running experiments.

<p align="center">
  <a href="https://arxiv.org/abs/2601.01392">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2601.01392-b31b1b.svg"/>
  </a>
  <a href="https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud">
    <img alt="Hugging Face Dataset" src="https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg"/>
  </a>
  <a href="https://www.modelscope.cn/datasets/JimmyMa99/TeleAntiFraud">
    <img alt="ModelScope Dataset" src="https://img.shields.io/badge/ModelScope-Dataset-orange.svg"/>
  </a>
  <a href="https://github.com/JimmyMa99/TeleAntiFraud">
    <img alt="TeleAntiFraud Repository" src="https://img.shields.io/badge/GitHub-TeleAntiFraud-black.svg"/>
  </a>
</p>

---

## Overview

This repository contains the source code for SAFE-QAQ, which consists of three main stages:
1. **Rule-Based Reinforcement Learning (Stage 1)**: Train a rule-based RL model.
2. **Rejection Sampling Fine-Tuning (RSFT) and Length-Constrained Reinforcement Learning (LCRL) (Stage 2)**: Refine the model using rejection sampling and LCRL techniques.
3. **Real-Time Fine-Tuning (Stage 3)**: Fine-tune the model for real-time inference.

The prompts for both real-time inference and training are defined in `prompt.py`.

## Resources

- [Paper (arXiv)](https://arxiv.org/abs/2601.01392)
- [TeleAntiFraud public dataset on Hugging Face](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud)
- [TeleAntiFraud public dataset on ModelScope](https://www.modelscope.cn/datasets/JimmyMa99/TeleAntiFraud)
- [TeleAntiFraud main repository](https://github.com/JimmyMa99/TeleAntiFraud)

SAFE-QAQ is trained and evaluated on the TeleAntiFraud audio-text fraud detection dataset. The companion TeleAntiFraud repository contains dataset resources, evaluation utilities, and additional project context.

---

## Environment Setup

To set up the environment, follow the instructions provided in [ms-swift](https://github.com/modelscope/ms-swift).

---

## Training and Inference Pipeline

### Stage 1: Rule-Based Reinforcement Learning
Train the initial rule-based RL model with:
```bash
bash run_swift_grpo_stage1.sh
```

### Stage 2: Rejection Sampling Fine-Tuning (RSFT) and Length-Constrained Reinforcement Learning (LCRL)
1. **Rejection Sampling**:
   Generate samples with:
   ```bash
   bash sample.sh
   ```
   Then process the sampled data with:
   ```bash
   bash process_samples.sh
   ```

2. **Fine-Tuning with RSFT**:
   Fine-tune the model on the processed data:
   ```bash
   bash run_swift_sft_stage2_RSFT.sh
   ```

3. **Length-Constrained Reinforcement Learning (LCRL)**:
   Further refine the model with LCRL:
   ```bash
   bash run_swift_grpo_stage2_LCRL.sh
   ```

### Stage 3: Real-Time Fine-Tuning
Run real-time fine-tuning with:
```bash
bash run_swift_grpo_stage3.sh
```

---

## Additional Notes
- The `prompt.py` file contains the definitions of prompts used during training and real-time inference.
- Ensure all dependencies are installed as per the [ms-swift](https://github.com/modelscope/ms-swift) documentation before running the scripts.
