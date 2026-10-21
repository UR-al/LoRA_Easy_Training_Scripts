# Anima-3.8B v1.1 LoRA (LoKr + DoRA, Qwen3.5-4B + Semantic Connector v2) training (example). Run from the repo root:
#   .\examples\anima\train_anima38.ps1 <dataset_config.toml> <output_name> [extra trainer args, e.g. --max_train_epochs=4]
# Settings: examples\anima\config_anima38.toml. Model files: examples\anima\models.ps1 (see README.md).
# Output: output\ANIMA\<output_name>, logs in output\ANIMA\log\<output_name>.
# Every argument after the two above goes to the trainer ("--key=value" and "--key value" both work).
# In PowerShell, quote a name that starts with @ ('@my_lora') and any value that contains a comma.
. "$PSScriptRoot\models.ps1"
. "$PSScriptRoot\anima_common.ps1"
Invoke-AnimaTraining -Model "3.8b" -Arguments $args
