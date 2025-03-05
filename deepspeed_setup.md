## Objective
This document goes over the setup process for using [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) alongside HuggingFace's [accelerate library](https://huggingface.co/docs/accelerate/en/index) for multi-GPU language model loading, training, and inference on the Prometheus cluster. 

DeepSpeed, an optimization library built on top of PyTorch, implements several stages of the [Zero Redundancy Optimizer (ZeRO) method](https://arxiv.org/abs/1910.02054) for efficient loading and training of deep learning models on GPUs. 

Accelerate, built by HuggingFace on top of PyTorch, is meant to increase the interoperability of PyTorch code across any distributed configuration.

_(The following procedure was tested on __Rocky Linux 8.5 x86_64__)._

## Environment Requirements
You should first install Anaconda. Please refer to [this documentation](https://github.com/rpattichis/deep_learning_dev_environment/blob/master/dl_env_setup.md#setup-conda) on how to set up Anaconda on the cluster.

Download the deepspeed-requirements.txt to the desired location in the cluster.

Once you've navigated to the directory where the deepspeed-requirements.txt file is located, create a new conda environment with the necessary installations with the following command line:

```
conda create -n env_name python=3.10
conda activate env_name

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install transformers 
pip install trl peft ninja
pip install flash-attn --no-build-isolation
pip install deepspeed sentencepiece
conda install -c conda-forge gcc=9 gxx=9
pip install protobuf
```

## Setting up DeepSpeed with accelerate

To create the appropriate Deepspeed config file, type ``` accelerate config ``` and answer the questions asked. Below are important ones with my personal notes:

#### _Which type of machine are you using?_
Select multi-GPU when it is required; make sure to verify your choice against the cluster specs.
#### _How many different machines will you use (use more than 1 for multi-node training)?_
Note that multi-node will take longer when running.
#### _Do you wish to optimize your script with torch dynamo?_
Optional; this basically calls torch.compile on your code before running but requires you to specify the compiler you want to use.
#### _Do you want to use DeepSpeed?_
Answer: yes!
#### _Do you want to specify a json file to a DeepSpeed config?_
This is optional, but I choose no because it will output a yaml file that you can edit later anyways.
#### _What should be your DeepSpeed's ZeRO optimization stage?_
This is super important. To understand your options, you can refer to this [documentation](https://www.deepspeed.ai/tutorials/zero/). Basically:
- Stage 1: Splits the optimizer states across processes.
- Stage 2: Splits optimizer and gradients across processes.
- Stage 3: Splits optimizer, gradient, and model weights across processes. Also includes CPU offload option.

If you're training a model, you probably want Stage 1/2; for inference, use Stage 3. Note that if Stage 1 works for you, prefer that, as more partitioning across processes will slow down your computation at run time.

Once you've answered the questions, it will tell you where the default_config.yaml was saved. It will probably be something like: `/trinity/home/username/.cache/huggingface/accelerate/`. If you ever need to modify the setup, you can either edit the file directly, or start this section from the beginning.

## Running DeepSpeed with accelerate

accelerate allows for a straight-forward integration of distributed training with DeepSpeed. Simply run the following line

```accelerate launch train_and_save_ex.py --args```

This will automatically look for the configuration file you've previously created. If instead, you prefer to have different configurations for different code (vs. a universal configuration as above) you can create a copy of the configuration file above, modify as necessary, move it to the same directory as your .py file, and then run the following:

```accelerate launch --config_file "deepspeed_config.yaml" train_and_save_ex.py --args```

#### Training Example with DeepSpeed + accelerate

Reference the ``train_and_save_ex.py`` for example code on how to run distributed loading, training, and saving of a large language model hosted publicly through Hugging Face.

__When saving a trained language model:__
If you're using ZeRO stages 1 or 2, nothing needs to be changed. However, when using Stage 3, you have to manually update the config yaml file and set the `zero3_save_16bit_model` to `true`. The uploaded config file example shows the configuration required for ZeRO-3 loading, training, and saving.

#### Inference Example with DeepSpeed + accelerate

_In progress, check in later!_

## FAQ

<details>
<summary>What's the difference between HuggingFace's accelerate and Transformers packages?</summary>
<br>
There are technically two documentations for using DeepSpeed, one for using Hugging Face's accelerate and one for using the Transformers packages. Under the hood, Transformers' integration actually relies on accelerate's integration.
</details>

<details>
<summary>How do I determine if I need multiple GPUs to load and/or train my language model?</summary>
<br>
Hugging Face provides a [neat resource](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator) that gives you an estimate of the amount of memory required for loading and training your model based on the precision of your weights, and training using different optimizers. Importantly, it also provides the memory requirement for the largest layer of the model. 
</details>

<details>
<summary>How can I see how much memory DeepSpeed placed on different GPUs?</summary>
<br>
After submitting a job and getting its jobID, you can run an interactive job using the jobID: `srun --jobid=123456 nvidia-smi`.
</details>

*Written by Rebecca Pattichis and tested on 4 March, 2025.* 