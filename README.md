# PneumoLLM: Harnessing the Power of Large Language Model for Pneumoconiosis Diagnosis

Updated on 2023.12.06


## Key Features

This repository provides the official implementation of *PneumoLLM: Harnessing the Power of Large Language Model for Pneumoconiosis Diagnosis*.

- New paradigm in applying large language models to diagnose data-scarce occupational diseases
- The novel contextual multi-token engine and information emitter module to meticulously draw out knowledge from LLMs
- Superiority in diagnosing pneumoconiosi, effectiveness of each designed module


## Links
- [Paper](https://arxiv.org/abs/2312.03490)
- [Code](https://github.com/CodeMonsterPHD/PneumoLLM/tree/main)


### Preparation
- Download [LLaMA-7B](https://huggingface.co/nyanko7/LLaMA-7B/tree/main) from HuggingFace (unofficial).

## Get Started

**Training**

- Before training, please modify related parameters, e.g., exp_name, and check the related parameters, e.g., epochs, lr, batch_size, accum_iter.
- Replace the data file and llama model path

```bash
--data_root
--llama_model_path
```

```bash
run train_acc.py
```

**Validation**

- Before validation, please modify related parameters.

```bash
run eval_acc.py
```

[**Checkpoint**][Baidu Drive](https://pan.baidu.com/s/1KSGlFn-GGFAyXC6DMgVbJA?pwd=chvt) or [Google Drive](https://drive.google.com/file/d/102oi317pqU8jhXJc9EW10WzXhhooHmAt/view?usp=sharing) 


## 📝 Citation

```text
@article{song2023pneumollm,
  title={PneumoLLM: Harnessing the Power of Large Language Model for Pneumoconiosis Diagnosis},
  author={Song, Meiyue and Yu, Zhihua and Wang, Jiaxin and Wang, Jiarui and Lu, Yuting and Li, Baicun and Wang, Xiaoxu and Huang, Qinghua and Li, Zhijun and Kanellakis, Nikolaos I and others},
  journal={arXiv preprint arXiv:2312.03490},
  year={2023}
}
```


