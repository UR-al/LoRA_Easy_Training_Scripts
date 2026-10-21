# Anima base 1.0 full fine-tune (anima_train.py) (example, experimental: see config_anima_fft.toml). Run from the repo root:
#   .\examples\anima\train_anima_fft.ps1 <dataset_config.toml> <output_name> [extra trainer args, e.g. --max_train_epochs=4]
# Settings: examples\anima\config_anima_fft.toml. Model files: examples\anima\models.ps1 (see README.md).
# Output: output\ANIMA\<output_name>, logs in output\ANIMA\log\<output_name>.
# Every argument after the two above goes to the trainer ("--key=value" and "--key value" both work).
# In PowerShell, quote a name that starts with @ ('@my_lora') and any value that contains a comma.
. "$PSScriptRoot\models.ps1"
. "$PSScriptRoot\anima_common.ps1"
Invoke-AnimaTraining -Model "fft" -Arguments $args
