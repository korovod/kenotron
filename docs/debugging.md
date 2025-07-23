# Debugging

## Debugging with VSCode

To debug with VSCode, add the following configuration to your `launch.json` file:

```json
{
    "name": "run_train.py",
    "type": "python",
    "request": "launch",
    "program": "torchrun", // or full path to torchrun by running `which torchrun`
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "--nproc_per_node=2",
        "run_train.py",
        "--config-file=examples/llama/config_tiny_llama.yaml", // or use examples/config_tiny_llama.py to generate your own config
    ],
    "env": {
        // "NANOTRON_BENCHMARK": "1", // enable to benchmark your training for a couple of steps
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "WANDB_MODE": "disabled",
    }
},
```
> [!NOTE]
> For more info check [Debugging Kénotron example (on multiple GPUs)](/examples/contributor-guide/README.md#debugging-nanotron-example-on-multiple-gpus)