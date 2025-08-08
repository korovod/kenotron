<h1 align="center">⚡️ Kénotron</h1>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> •
        <a href="#quick-start">Quick Start</a> •
        <a href="#features">Features</a> •
        <a href="CONTRIBUTING.md">Contributing</a>
    <p>
</h4>

<h3 align="center">
    <p>Pretraining models made easy</p>
</h3>


Kénotron is a library for pretraining transformer models. It provides a simple and flexible API to pretrain models on custom datasets. Kénotron is designed to be easy to use, fast, and scalable. It is built with the following principles in mind:

- **Simplicity**: Kénotron is designed to be easy to use. It provides a simple and flexible API to pretrain models on custom datasets.
- **Scalability**: Kénotron uses the latest techniques to train models more efficiently at scale.
- **Speed**: This version of Nanotron focuses on HPC-oriented optimizations, typically made available via C++ extensions.

## Installation

We recommend using [Spack](https://spack.io/) to install Kénotron.

```bash
git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git
cd spack/bin
./spack repo add --name korovod https://github.com/korovod/korovod-spack-packages.git
./spack install py-nanotron
```

Spack allows you to install a specific version e.g., `py-nanotron@0.4.0` or `py-nanotron@main`.

> [!TIP]
> It is advised to maintain a proper [Spack environment](https://spack-tutorial.readthedocs.io/en/latest/tutorial_environments.html) to ensure reproducibility.

To install an extension, simply use the corresponding Spack variant:

```bash
./spack install py-nanotron +datastates +nanosets
```

Some examples are shipped into Kénotron, with some of them requiring the installation of a Spack variant:

| Variant | Description | Docs | Spack variant |
| --- | --- | --- | --- |
| `datastates` | Asynchronous checkpointing | [Docs](/examples/datastates/README.md) | `py-nanotron +datastates` |
| `nanosets` | Use the datatrove library to load data | [Docs](./examples/nanosets/README.md)  | `py-nanotron +nanosets` |
| `custom-dataloader` | Plug a custom dataloader to Kénotron | [Docs](./examples/custom-dataloader/README.md) | `py-nanotron` |
| `doremi` | Use DoReMi to speed up training | [Docs](./examples/doremi/README.md) | `py-nanotron` |
| `mamba` | Train an example Mamba model | [Docs](./examples/mamba/README.md) | `py-nanotron` |
| `moe` | Train an example Mixture-of-Experts (MoE) model | [Docs](./examples/moe/README.md) | `py-nanotron` |
| `mup` | Use spectral µTransfer to scale up your model | [Docs](./examples/mup/README.md) | `py-nanotron` |
| `s3` | For automatically uploading checkpoints to S3 | [Docs](./examples/s3/README.md) | `py-nanotron` |

## Quick Start

First, have a look at the **[Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)**, a comprehensive guide to efficiently scale LLM training with Nanotron. Everything in this guide applies to Kénotron.

### Predicting the memory that you will need

A good starting point is to understand the memory usage from model configurations. The Nanotron team created [a tool](https://huggingface.co/spaces/nanotron/predict_memory) for this purpose. Just paste your YAML configuration to generate memory diagrams.

### Training a tiny Llama model

The following command will train a tiny Llama model on a single node with 8 GPUs. The model will be saved in the `checkpoints` directory as specified in the config file.

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 python -m torch.distributed.run --nproc_per_node=8 run_train.py --config-file examples/llama/config_tiny_llama.yaml
```

For detailed instructions on training your first model, check out our [Your First Training guide](docs/your-first-training.md).

For multi-node training with Slurm, see our [Multi-Node Training guide](docs/multi-node-training.md).

### Run generation from your checkpoint

```bash
python -m torch.distributed.run --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/10/ --tp 1 --pp 1
# We could set a larger TP for faster generation, and a larger PP in case of very large models.
```

## Features

We currently support the following features:

- [x] 3D parallelism (DP+TP+PP)
- [x] Expert parallelism for MoEs
- [x] AFAB and 1F1B schedules for PP
- [x] Explicit APIs for TP and PP which enables easy debugging
- [x] ZeRO-1 optimizer
- [x] FP32 gradient accumulation
- [x] Parameter tying/sharding
- [x] Custom module checkpointing for large models
- [x] Spectral µTransfer parametrization for scaling up neural networks
- [x] Mamba example
- [x] Asynchronous checkpointing

And we have on our roadmap:

- [ ] Data-parallel checkpointing for reducing I/O pressure
- [ ] FSDP
- [ ] `torch.compile` support
- [ ] Interleaved 1f1b schedule
- [ ] Efficient expert parallelism

## Models

The following models are currently supported:

- Mistral 7B
- Qwen
- Llama 3.2
- Llama 3.1
- StarCoder2

## Credits

We thank the Hugging Face team for their work on the [original project](https://github.com/huggingface/nanotron).

Some related projects are:

- [gpt-neox](https://github.com/EleutherAI/gpt-neox)
- [torchtitan](https://github.com/pytorch/torchtitan)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [lingua](https://github.com/facebookresearch/lingua)
