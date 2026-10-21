# MOSS-Image LoRA training (example). Run from the repo root:
#   .\examples\moss\train_moss.ps1 <dataset_config.toml> <output_name> [extra args, e.g. --max_train_epochs=4]
$arg0 = $args[0] # dataset config TOML
$arg1 = $args[1] # output LoRA name
$network_args = @()
$base_output_path = ".\output\MOSS"

# every argument after the two above is passed through unchanged ("--key=value" and "--key value" both work)
if ($args.Count -gt 2) {
    $network_args = $args[2..($args.Count - 1)]
}

# UTF-8 output: the trainer prints Japanese / Korean text, which fails in the ANSI code page when stdout is redirected
$env:PYTHONIOENCODING = "utf-8"
.\backend\sd_scripts\venv\Scripts\Activate.ps1

accelerate launch `
--dynamo_backend no `
--dynamo_mode default `
--mixed_precision bf16 `
--num_processes 1 `
--num_machines 1 `
--num_cpu_threads_per_process 5 `
".\backend\sd_scripts\moss_train_network.py" `
--config_file "$PSScriptRoot\config_moss.toml" `
--dataset_config "$arg0" `
--output_dir "$base_output_path\$arg1" `
--logging_dir "$base_output_path\log\$arg1" `
--output_name="$arg1" `
@network_args
