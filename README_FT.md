## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```
## Related Code Structure

- Model: ``./models/modeling_llama.py``, modified from same file in package ``transformers==4.25.1``. Mainly added parameters contains zs part, for example in line 1660:
``` diff
     def forward(
         self,
         input_ids: torch.LongTensor = None,
         attention_mask: Optional[torch.Tensor] = None,
         position_ids: Optional[torch.LongTensor] = None,
         past_key_values: Optional[List[torch.FloatTensor]] = None,
         inputs_embeds: Optional[torch.FloatTensor] = None,
         labels: Optional[torch.LongTensor] = None,
         use_cache: Optional[bool] = None,
         output_attentions: Optional[bool] = None,
         output_hidden_states: Optional[bool] = None,
         return_dict: Optional[bool] = None,
+        head_z = None,
+        head_layer_z = None,
+        intermediate_z = None,
+        mlp_z = None,
+        hidden_z = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
```

- Lora: ``./utils/lora_utils.py``

- Dataloader: ``./tasks/alpaca.py``

- Datasets: we used two datasets for Finetuning, [alpaca-gpt4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) and [alpaca-cleaned](https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json)

## Finetuning(train.py)

- Runing Finetune by our pipeline: Finetune is the second step in our pruning-finetuning pipeline. To run finetune, I will provide our pruning results as a start baseline for finetune, consisting of a lora weights file (``lora_weights.pt``) and a structured mask file (``zs.pt``). Here is an example to run Finetune: ``./scripts/finetune/finetune_lr8e-6_cosine_FTepoch2_mark25_gpt4alpaca_promptlong.sh``

  - Step 1: Save the lora weights ``lora_weights.pt`` and mask file ``zs.pt`` to a folder, and replace ``baseline_pruned_model`` to the new folder path;

  - Step 2: Prepare the lora weights and merge lora weights (sent by OneDrive) back to origin model (decapoda-research/llama-7b-hf) by ``merge_weights.py`` (Step 3 provides an example) and get a new model as our baseline model in FT;

  - Step 3: run finetuning scripts.

## Evaluation

Evaluate on wikitext2 dataset and get the ppl metric:

Before Finetune:
```bash
export PYTHONPATH='.'

python ./eval_ppl/eval_ppl.py \
    --max_seq_len 1024 \
    --model_type lora_pruner \
    --base_model ./llama_pruned \
    --lora_ckpt $pretrained_pruned_model \
    --lora_merged
```

Notice that if set ``lora_merged``, lora weights in ``lora_ckpt`` will not be added to ``base_model``.

After Finetune:
```bash
export PYTHONPATH='.'
cp $pretrained_pruned_model/zs.pt $finetuned_lora_dir

python ./eval_ppl/eval_ppl.py \
    --max_seq_len 1024 \
    --model_type lora_pruner \
    --base_model ./llama_pruned \
    --lora_ckpt $finetuned_lora_dir \
```

