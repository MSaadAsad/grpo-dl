# Assignment 1: Initial Draft

## 📚 Description

This project explores two applications of reinforcement learning and fine-tuning:

- **DL_Bipedal_Walker.ipynb** – Applies PPO and GRPO to train a policy for continuous control in the BipedalWalker-v3 environment.
- **GPT_2_Finetune.ipynb** – Fine-tunes a GPT-2 model on structured math reasoning tasks using the GSM8K dataset with SFT and GRPO.

Both notebooks run and produce initial results. Further tuning, evaluation, and comparisons are in progress.

---

## 📁 Files

- `DL_Bipedal_Walker.ipynb` – PPO vs GRPO on a continuous control task.
- `GPT_2_Finetune.ipynb` – GPT-2 fine-tuning on math reasoning.
- `initial_report.pdf` – Method overview, experimental setup, and planned improvements.

---

## 🚀 How to Run

Just open the notebooks in Jupyter and run the cells.

Make sure the required libraries are installed:
- `torch`
- `transformers`
- `datasets`
- `gym`
- `wandb`
- `pandas`
- `matplotlib`

---

## ✅ Current Status

- ✅ Both notebooks execute successfully.
- ⚠️ PPO for GPT-2 is not yet implemented.
- ⚠️ GRPO uses small batch size (batch = 1, 4–8 generations per prompt) due to resource constraints.
- 🛠️ Evaluation and logging tools still being improved.

---

## 🔜 Next Steps

- Implement PPO for GPT-2 and compare with GRPO.
- Scale GRPO to the full GSM8K dataset with better batch parallelism.
- Add deployment via REST API.
- Improve visualizations using Weights & Biases (WandB).
