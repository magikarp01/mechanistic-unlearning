# mechanistic-unlearning
Research for unlearning concepts and capabilities using mechanistic interpretability understanding/localization techniques.

Setup:
Python 3.10.13
- Pytorch 11.8 with conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
- transformers
mask_learning.py will download GPT2-medium in huggingface transformers library

## Tasks
Masks should be trained using "Task" objects. Task 

# Circuit Breaking: Removing Model Behaviors with Targeted Ablation

The repository is split into `mnist` and `toxicity` folders, corresponding to the two experimental settings described in the paper.

For the `toxicity` setting, look in the `toxicity` folder:
- `toxic_data_for_eval.py` extracts toxic samples from the [4chan dataset](https://arxiv.org/abs/2001.07487) and stores them in the `data` folder. 
- `compute_means.py` computes the mean of the GPT-2 embeddings for a [10k sample of OpenWebText](https://huggingface.co/datasets/NeelNanda/pile-10k).
- `evaluation.py` evaluates the original, ablated, and fine-tuned model on the OWT dataset.
- `finetune_gpt2.py` finetunes GPT-2 against toxic comments, using eq. 4 from the paper.
- `train_mask.py` trains a binary mask on the GPT2 model to ablate edges in the graph, implementing targeted ablation per Section 3 of the paper.
- `transformer.py` implements a modified version of the transformer architecture to enable casaul path interventions.
- `utils.py` provides utilities for inference with toxic and OWT data.

For the `mnist` setting, look in the `mnist` folder:
- You can run `main.py` to run the ablation experiment
- The `data` folder has the `MNIST` images to train on
- The `mlp_model.py` file defines the architecture we use for these experiments
- `old.py` has an old version of `main.py`


Generate IOI data in make_ioi_datasets.ipynb.
Run compute_means.py for specific or set means=False for zero ablation.
Train masks in train_ioi_mask.ipynb or alternative_masks.ipynb.1


interactive mode command: srun --nodes=1 --cpus-per-task=4 --gres=gpu:1 --time=1:00:00 --pty bash -i