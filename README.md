# LLM Module: Train and evaluate LLM modularly

A tool with multiple modular components for the training and evaluation of LLM

Purpose of Development: There are two types of tools in the market.

One is user-friendly with web UI for easy access, but offers limited operational space and lacks customization.

The other type is code-based with strong operability, yet comes with a high learning curve and coding requirements.

Hence, this tool was developed to operate without the need for coding, solely through configuration files, akin to
Java's Spring.

It allows for swift mastery of large models and offers precise control at the most granular level, making model
customization remarkably convenient.

[![GitHub Repo stars](https://img.shields.io/github/stars/lessonnair/llm-module?style=social)](https://github.com/lessonnair/llm-module/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/lessonnair/llm-module)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/lessonnair/llm-module)](https://github.com/lessonnair/llm-module/commits/main)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/lessonnair/llm-module/pulls)

\[ English | [中文](README_cn.md) \]

## Quick Start

1. Install dependencies

```shell
pip install -r requirements.txt
```

2. Write a configuration file

In this file, you need to define a `pipeline`.

The `pipeline` represents a workflow that sequentially runs associated components in the defined order.

The components used in it need to be pre-defined in the context.

```config
[Project]
name=myLLM
version=1.0
user=test
proxies={"http": "127.0.0.1:7890", "https": "127.0.0.1:7890"}

# The pipeline represents a predefined workflow composed of various components in the following context.
pipeline=Trainer,Export,Chat

[TokenizerLoader]
# You can configure all the parameters that appear in the methods `transformers.AutoConfig.from_pretrained`
and `transformers.AutoTokenizer.from_pretrained`.
pretrained_model_name_or_path=THUDM/chatglm2-6b
use_fast=True
;split_special_tokens=True
;padding_side=right
device_map=auto
trust_remote_code=True

[ModelLoader]
# You can specify additional parameters for the class below, for example, `transformers.AutoModelForCausalLM`
class=transformers.AutoModel
pretrained_model_name_or_path=THUDM/chatglm2-6b
print_model_structure=False
trust_remote_code=True
cache_dir=./cache
device_map=auto
use_auth_token=False
;torch_dtype=bf16
use_gradient_checkpointing=False

# for LLama and Falcon models
;rope_scaling=dynamic
model_max_length=2000
flash_attn=False
shift_attn=False
;quantization_bit=8
;double_quantization=4
quantization_type=
finetune_args=FinetuneArguments

[FinetuneArguments]
# type can be full,freeze or lora
type=lora
checkpoint_dir=

# you can set parameters available in the peft.LoraConfig method, make sure to prefix the parameter names with lora_config, for example, `lora_config_target_modules` instead of `target_modules`.
lora_config_task_type=CAUSAL_LM
lora_config_inference_mode=False
lora_config_r=4
lora_config_lora_alpha=32
lora_config_lora_dropout=0.05
lora_config_target_modules=query_key_value
# bias can be set none or all or lora_only
;lora_config_bias=none
;lora_config_fan_in_fan_out=True

;num_layer_trainable=2
;upcast_layernorm=True
;neft_alpha=1e-6

[TrainingArguments]
# You can specify additional parameters for the class below, for example, `transformers.Seq2SeqTrainingArguments`
; class=transformers.TrainingArguments
class=transformers.Seq2SeqTrainingArguments
generation_max_length=256
generation_num_beams=1
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=2
warmup_steps=20
max_steps=800
;num_train_epochs=1
learning_rate=1e-5
;fp16=True
logging_steps=25
do_train=True
remove_unused_columns=False
output_dir=./output_train
save_safetensors=False
seed=2023

[GenerateArguments]
# you can set parameters available in the transformers.GenerationConfig method
do_sample=True
temperature=1.0
;top_p=1.0
;top_k=50
num_beams=1
max_length=2048
;max_new_tokens=128
repetition_penalty=1.0
length_penalty=1.0

[Trainer]
class=transformers.Seq2SeqTrainer
# The previously defined ModelLoader is referenced here.
model=ModelLoader
# The previously defined TokenizerLoader is referenced here.
tokenizer=TokenizerLoader
# The previously defined TrainingArguments is referenced here.
args=TrainingArguments
# The previously defined DatasetLoader is referenced here.
dataset=DatasetLoader_1
resume_from_checkpoint=False
plot_loss=True
ignore_pad_token_for_loss=True
streaming=False
split_train_val=True
split_train_val_val_size=20
split_train_val_seed=2024
split_train_val_buffer_size=10
steps=train,eval,predict
predict_with_generate=True
ppo_args=PPOArguments
generate_args=GenerateArguments

# stage can be pt,sft,ppo,dpo or rm
stage=sft
# when stage is ppo, you need to set reward_model
reward_model=./output_reward_model

[DatasetLoader_1]
# type should be hf_hub or script or file
path=json
data_files=./data/oaast_sft_zh.json
text_column=instruction
prompt_column=instruction
query_column=input
history_column=history
response_column=output
system_column=system
split=train
cache_dir=./cache
streaming=False
use_auth_token=False

# etl process params
# The previously defined TokenizerLoader is referenced here.
tokenizer=TokenizerLoader
# stage can be pt or sft or rm or ppo
cutoff_len=128
sft_packing=True

# Rendering types for prompts,different value is required for different LLM
render=chatglm2
label_mask_prompt=True

[Export]
# The TokenizerLoader defined above is referenced.
tokenizer=TokenizerLoader
# The ModelLoader defined above is referenced.
model=ModelLoader
output_dir=./export/chatglm2_lora
max_shard_size=5gb

[Chat]
tokenizer=TokenizerLoader
model=ModelLoader
pretrained_model_name_or_path=./export/chatglm2_lora
generating_args=GenerateArguments
render=chatglm2
```

3. Run `sh run.sh example.conf`

- example.conf is the task configuration file defined above

## Supported Component

### 1. How to define a component

In the configuration file, a `section` represents a component, formatted as

```config
[ComponentName]
property1=value1
property2=value2
...
```

### 2. How to name a component

- Direct use of component type

For example, `Trainer`, `TokenizerLoader`, etc.

- Concatenate component type with a suffix

For example, `DatasetLoader_1`, `Trainer_ppo`, etc.
The system will automatically recognize its type based on the
prefix and then obtain its instance automatically.

### 3. Currently supported component types

- Trainer
- DatasetLoader
- TokenizerLoader
- ModelLoader
- Export
- Chat
- TrainingArguments
- GenerateArguments
- FinetuneArguments

## Supported Models

| Model                                                    | Model size                  | target_modules    | Render    |
| -------------------------------------------------------- | --------------------------- | --------- |-----------|
| [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B              | q_proj,v_proj | -         |
| [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B                  | q_proj,v_proj | llama2    |
| [BLOOM](https://huggingface.co/bigscience/bloom)         | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value | -         |
| [BLOOMZ](https://huggingface.co/bigscience/bloomz)       | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value | -         |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b)        | 7B/40B                      | query_key_value | -         |
| [Baichuan](https://github.com/baichuan-inc/Baichuan-13B) | 7B/13B                      | W_pack    | baichuan  |
| [Baichuan2](https://github.com/baichuan-inc/Baichuan2)   | 7B/13B                      | W_pack    | baichuan2 |
| [InternLM](https://github.com/InternLM/InternLM)         | 7B/20B                      | q_proj,v_proj | intern    |
| [Qwen](https://github.com/QwenLM/Qwen-7B)                | 7B/14B                      | c_attn    | chatml    |
| [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)         | 6B                          | query_key_value | chatglm2  |
| [Phi-1.5](https://huggingface.co/microsoft/phi-1_5)      | 1.3B                        | Wqkv      | -         |

## Supported Training Approaches

| Approach               |   Full-parameter   | Partial-parameter  |       LoRA         |       QLoRA        |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Pre-Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Supervised Fine-Tuning | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Reward Modeling        |                    |                    | :white_check_mark: | :white_check_mark: |
| PPO Training           |                    |                    | :white_check_mark: | :white_check_mark: |
| DPO Training           | :white_check_mark: |                    | :white_check_mark: | :white_check_mark: |

## References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)