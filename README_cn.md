# LLM Module: 以模块组件化的方式玩转大模型

**一个模块化工具，用来训练、评估大模型。**

开发初衷：市面上的工具分为两类

一类是零门槛入手，webui页面操作，但可操作空间小，不方便定制化。

另一类是代码类工具，虽然操作性强，但上手门槛高，有编写代码成本。

故而开发了这个工具，无需代码，仅通过配置文件的方式，
类似Java里面的Spring。

既可以最快玩转大模型，又能在最细微的层面控制大模型，非常方便定制化模型。

[![GitHub Repo stars](https://img.shields.io/github/stars/lessonnair/llm-module?style=social)](https://github.com/lessonnair/llm-module/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/lessonnair/llm-module)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/lessonnair/llm-module)](https://github.com/lessonnair/llm-module/commits/main)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/lessonnair/llm-module/pulls)

\[ 中文 | [English](README.md) \]

## 快速上手

1. 安装依赖

```shell
pip install -r requirements.txt
```

2. 编写配置文件

在这个文件中，你需要定义 `pipeline`。

`pipeline`代表了一个工作流，它会按照定义的顺序逐一运行相关的组件。

其中，使用到的组件需要在上下文中定义好。

```config
[Project]
name=myLLM
version=1.0
user=test
proxies={"http": "127.0.0.1:7890", "https": "127.0.0.1:7890"}

# pipeline代表了定义好的工作流，有下文的各个组件组成
pipeline=Trainer,Export,Chat

[TokenizerLoader]
# 你可以设置出现在 `transformers.AutoConfig.from_pretrained` 和 `transformers.AutoTokenizer.from_pretrained`中所有的参数
pretrained_model_name_or_path=THUDM/chatglm2-6b
use_fast=True
;split_special_tokens=True
;padding_side=right
device_map=auto
trust_remote_code=True

[ModelLoader]
# 你可以设置下面定义的class的参数，例如，`transformers.AutoModel`初始化的所有参数
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
# type 可以设置为 full、freeze或者lora
type=lora
checkpoint_dir=

# 你可以设置 `peft.LoraConfig`方法中的参数，需要注意，参数需要加上 lora_config的前缀。例如，需要使用 `lora_config_target_modules` 来替代 `target_modules`
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
# 你可以设置下面定义的class的参数，例如，`transformers.Seq2SeqTrainingArguments`的所有参数
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
# 你可以设置 transformers.GenerationConfig 的所有参数
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
# 这里引用了上文定义的ModelLoader
model=ModelLoader  
# 这里引用了上文定义的TokenizerLoader
tokenizer=TokenizerLoader
# 这里引用了上文定义的TrainingArguments
args=TrainingArguments
# 这里引用了上文定义的 DatasetLoader
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

# stage 可以设置为 pt,sft,ppo,dpo 或者 rm
stage=sft
# 当stage设置为ppo时，需要同时设置reward_model，且确保改路径下有相应的模型
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
# 这里引用了上文定义的 TokenizerLoader
tokenizer=TokenizerLoader
cutoff_len=128
sft_packing=True
# prompt渲染类型，不同llm需要设置不同的值
render=chatglm2
label_mask_prompt=True

[Export]
# 引用了上文定义的 TokenizerLoader
tokenizer=TokenizerLoader
# 引用了上文定义的 ModelLoader
model=ModelLoader
# 模型导出路径
output_dir=./export/chatglm2_lora
max_shard_size=5gb

[Chat]
tokenizer=TokenizerLoader
model=ModelLoader
pretrained_model_name_or_path=./export/chatglm2_lora
generating_args=GenerateArguments
render=chatglm2
```

3. 运行 `sh run.sh example.conf`

- `example.conf` 是第2步定义的任务配置文件

## 目前支持的组件

### 1、组件概念

配置文件中一个 `section` 代表一个组件，格式为

```config 
[组件名]
属性1=值1
属性2=值2
...
```

### 2、组件名的规则

- 直接使用组件类型
  例如 `Trainer`， `TokenizerLoader` 等
- 组件类型_后缀
  例如 `DatasetLoader_1`, `Trainer_ppo`等，系统会通过前缀自动识别它的类型，进而获取它的实例

### 3、组件类型

- Trainer
- DatasetLoader
- TokenizerLoader
- ModelLoader
- Export
- Chat

- TrainingArguments
- GenerateArguments
- FinetuneArguments

## 目前支持的模型

| 模型名称                                                     | 模型大小                        |  target_modules    | render    |
|----------------------------------------------------------|-----------------------------| ----------------- |-----------|
| [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B              | q_proj,v_proj     | -         |
| [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B                  | q_proj,v_proj     | llama2    |
| [BLOOM](https://huggingface.co/bigscience/bloom)         | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -         |
| [BLOOMZ](https://huggingface.co/bigscience/bloomz)       | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -         |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b)        | 7B/40B                      | query_key_value   | -         |
| [Baichuan](https://github.com/baichuan-inc/Baichuan-13B) | 7B/13B                      | W_pack            | baichuan  |
| [Baichuan2](https://github.com/baichuan-inc/Baichuan2)   | 7B/13B                      | W_pack            | baichuan2 |
| [InternLM](https://github.com/InternLM/InternLM)         | 7B/20B                      | q_proj,v_proj     | intern    |
| [Qwen](https://github.com/QwenLM/Qwen-7B)                | 7B/14B                      | c_attn            | chatml    |
| [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)         | 6B                          | query_key_value   | chatglm2  |
| [Phi-1.5](https://huggingface.co/microsoft/phi-1_5)      | 1.3B                        | Wqkv              | -         |

## 目前支持的训练模式

| Approach               |   Full-parameter   | Partial-parameter  |       LoRA         |       QLoRA        |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Pre-Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Supervised Fine-Tuning | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Reward Modeling        |                    |                    | :white_check_mark: | :white_check_mark: |
| PPO Training           |                    |                    | :white_check_mark: | :white_check_mark: |
| DPO Training           | :white_check_mark: |                    | :white_check_mark: | :white_check_mark: |

## 鸣谢

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)