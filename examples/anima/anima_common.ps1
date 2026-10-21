# Shared code of the examples\anima\train_anima*.ps1 launchers (dot-sourced after models.ps1). Not meant to be run directly.
# Checks that every file the chosen model needs exists BEFORE accelerate starts, then runs the trainer from the repo root.

$AnimaExampleDir = $PSScriptRoot

$AnimaHfBase = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files"
$Anima29Url = "https://huggingface.co/Gazingstars123/Anima-2.9B/resolve/main/Anima-2.9B-preview-v1.safetensors"
$Anima38Repo = "https://huggingface.co/lylogummy/Anima-3.8B/resolve/main"
$AnimaQwen35TokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

function New-AnimaRequirement([string]$Label, [string]$Path, [string]$Url, [string]$Note) {
    return @{ Label = $Label; Path = $Path; Url = $Url; Note = $Note }
}

function Get-AnimaModelSetup([string]$Model) {
    $qwen3 = New-AnimaRequirement "Qwen3-0.6B text encoder" ([System.IO.Path]::Combine($AnimaTextEncoderDir, $AnimaQwen3File)) `
        "$AnimaHfBase/text_encoders/qwen_3_06b_base.safetensors" "repo circlestone-labs/Anima"
    $vae = New-AnimaRequirement "Qwen-Image VAE" ([System.IO.Path]::Combine($AnimaVaeDir, $AnimaVaeFile)) `
        "$AnimaHfBase/vae/qwen_image_vae.safetensors" "repo circlestone-labs/Anima"
    $setup = @{ Script = "anima_train_network.py"; Threads = 5; Extra = @() }
    switch ($Model) {
        "base" {
            $setup.Launcher = "train_anima_base.ps1"; $setup.Config = "config_anima_base.toml"
            $dit = New-AnimaRequirement "Anima base 1.0 DiT" ([System.IO.Path]::Combine($AnimaDitDir, $AnimaBaseFile)) `
                "$AnimaHfBase/diffusion_models/anima-base-v1.0.safetensors" "repo circlestone-labs/Anima"
        }
        "fft" {
            $setup.Launcher = "train_anima_fft.ps1"; $setup.Config = "config_anima_fft.toml"
            $setup.Script = "anima_train.py"; $setup.Threads = 8
            $dit = New-AnimaRequirement "Anima base 1.0 DiT" ([System.IO.Path]::Combine($AnimaDitDir, $AnimaBaseFile)) `
                "$AnimaHfBase/diffusion_models/anima-base-v1.0.safetensors" "repo circlestone-labs/Anima"
        }
        "2.9b" {
            $setup.Launcher = "train_anima29.ps1"; $setup.Config = "config_anima29.toml"
            $dit = New-AnimaRequirement "Anima-2.9B preview DiT" ([System.IO.Path]::Combine($AnimaDitDir, $Anima29File)) `
                $Anima29Url "repo Gazingstars123/Anima-2.9B, no login needed; circlestone-labs non-commercial license, see LICENSE.md there"
        }
        "3.8b" {
            $setup.Launcher = "train_anima38.ps1"; $setup.Config = "config_anima38.toml"
            $dit = New-AnimaRequirement "Anima-3.8B v1.1 bundle (DiT + connector)" ([System.IO.Path]::Combine($AnimaDitDir, $Anima38File)) `
                "$Anima38Repo/difussion_models/Anima-3.8B-v1.1.safetensors" "repo lylogummy/Anima-3.8B, folder difussion_models"
        }
        default { throw "unknown Anima model '$Model' (base, 2.9b, 3.8b, fft)" }
    }
    $setup.Files = @($dit, $qwen3, $vae)
    $setup.ModelArgs = @(
        "--pretrained_model_name_or_path", $dit.Path,
        "--qwen3", $qwen3.Path,
        "--vae", $vae.Path
    )
    if ($Model -eq "3.8b") {
        $qwen35 = New-AnimaRequirement "Qwen3.5-4B text encoder" ([System.IO.Path]::Combine($AnimaTextEncoderDir, $AnimaQwen35File)) `
            "$Anima38Repo/text_encoders/qwen35_4b.safetensors" "repo lylogummy/Anima-3.8B, folder text_encoders"
        $tokDir = $AnimaQwen35TokenizerDir.TrimEnd("\", "/")
        $tokJson = New-AnimaRequirement "Qwen3.5 tokenizer.json" ([System.IO.Path]::Combine($tokDir, "tokenizer.json")) `
            "$AnimaQwen35TokenizerUrl/tokenizer.json" "repo Qwen/Qwen3.5-4B; only the two tokenizer files are needed"
        $tokCfg = New-AnimaRequirement "Qwen3.5 tokenizer_config.json" ([System.IO.Path]::Combine($tokDir, "tokenizer_config.json")) `
            "$AnimaQwen35TokenizerUrl/tokenizer_config.json" "repo Qwen/Qwen3.5-4B; only the two tokenizer files are needed"
        $setup.Files += @($qwen35, $tokJson, $tokCfg)
        $setup.ModelArgs += @("--qwen35_path", $qwen35.Path, "--qwen35_tokenizer_dir", $tokDir)
    }
    return $setup
}

# Returns $null when the file is UTF-8 without BOM (plain ASCII included), otherwise what is wrong with it.
function Test-AnimaUtf8NoBom([string]$Path) {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) { return "it starts with a UTF-8 BOM" }
    if ($bytes.Length -ge 2 -and (($bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) -or ($bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF))) {
        return "it is UTF-16 (Unicode)"
    }
    try {
        [void](New-Object System.Text.UTF8Encoding($false, $true)).GetString($bytes)
    } catch {
        return "it is not valid UTF-8 (saved in an ANSI code page?)"
    }
    return $null
}

function Invoke-AnimaTraining([string]$Model, [object[]]$Arguments) {
    if ($null -eq $Arguments) { $Arguments = @() }
    $setup = Get-AnimaModelSetup $Model
    $usage = ".\examples\anima\$($setup.Launcher) <dataset_config.toml> <output_name> [extra trainer args, e.g. --max_train_epochs=4]"

    if ($Arguments.Count -lt 2 -or -not $Arguments[0] -or -not $Arguments[1]) {
        Write-Host "usage: $usage" -ForegroundColor Yellow
        Write-Host "       In PowerShell, quote a name that starts with @ ('@my_lora') and any value that contains a comma." -ForegroundColor Yellow
        exit 1
    }
    $datasetArg = [string]$Arguments[0]
    $outputName = [string]$Arguments[1]
    # Passed through as typed: "--key=value" and "--key value" both work. An unquoted a,b reaches the script as an
    # array; it becomes separate arguments (as in a direct accelerate call) instead of one "a b" argument.
    $extraArgs = @()
    if ($Arguments.Count -gt 2) {
        $extraArgs = @(foreach ($a in $Arguments[2..($Arguments.Count - 1)]) { foreach ($v in @($a)) { [string]$v } })
    }

    $problems = 0
    if ($AnimaLocalModelsError) {
        Write-Host "[error] examples\anima\models.local.ps1 could not be read:" -ForegroundColor Red
        Write-Host "        $AnimaLocalModelsError" -ForegroundColor Red
        $problems++
    }
    $datasetConfig = $null
    try { $datasetConfig = ConvertTo-AnimaFullPath $datasetArg } catch { }  # relative to where you ran the launcher
    if (-not $datasetConfig -or -not (Test-Path -LiteralPath $datasetConfig -PathType Leaf)) {
        Write-Host "[missing] dataset config: $datasetArg" -ForegroundColor Red
        Write-Host "          copy examples\anima\dataset_example.toml (e.g. to datasets\my_style.toml), edit image_dir and pass its path" -ForegroundColor Red
        $problems++
    } else {
        $encodingProblem = Test-AnimaUtf8NoBom $datasetConfig
        if ($encodingProblem) {
            Write-Host "[error] dataset config: $datasetArg ($encodingProblem)" -ForegroundColor Red
            Write-Host "        save it as UTF-8 without BOM (Windows 11 Notepad: 'UTF-8', not 'UTF-8 with BOM')" -ForegroundColor Red
            $problems++
        }
    }
    if ($outputName.IndexOfAny([char[]]'\/:*?"<>|') -ge 0) {
        Write-Host "[error] output_name must be a plain name, not a path: $outputName" -ForegroundColor Red
        $problems++
    }

    $activate = Join-Path $AnimaRepoRoot "backend\sd_scripts\venv\Scripts\Activate.ps1"
    $accelerateExe = Join-Path $AnimaRepoRoot "backend\sd_scripts\venv\Scripts\accelerate.exe"
    $trainer = Join-Path $AnimaRepoRoot ("backend\sd_scripts\" + $setup.Script)
    $config = Join-Path $AnimaExampleDir $setup.Config
    if (-not (Test-Path -LiteralPath $trainer -PathType Leaf)) {
        Write-Host "[missing] trainer: $trainer" -ForegroundColor Red
        Write-Host "          run install.bat in the repo root (it fetches the backend\sd_scripts submodule)" -ForegroundColor Red
        $problems++
    }
    if (-not (Test-Path -LiteralPath $activate -PathType Leaf)) {
        Write-Host "[missing] training venv: $activate" -ForegroundColor Red
        Write-Host "          run install.bat in the repo root and answer y to 'Are you using this locally?'" -ForegroundColor Red
        $problems++
    } elseif (-not (Test-Path -LiteralPath $accelerateExe -PathType Leaf)) {
        Write-Host "[missing] accelerate: $accelerateExe" -ForegroundColor Red
        Write-Host "          the venv is incomplete (interrupted install?): re-run install.bat in the repo root" -ForegroundColor Red
        $problems++
    }
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
        Write-Host "[missing] config: $config" -ForegroundColor Red
        $problems++
    }

    $missingModels = 0
    foreach ($file in $setup.Files) {
        if (-not $file.Path -or -not (Test-Path -LiteralPath $file.Path -PathType Leaf)) {
            Write-Host "[missing] $($file.Label)" -ForegroundColor Red
            Write-Host "          expected: $($file.Path)" -ForegroundColor Red
            Write-Host "          download: $($file.Url)" -ForegroundColor Red
            Write-Host "                    ($($file.Note))" -ForegroundColor Red
            $problems++
            $missingModels++
        }
    }
    if ($problems -gt 0) {
        Write-Host ""
        Write-Host "Not started: $problems problem(s) above. Model folder: $AnimaModelRoot" -ForegroundColor Yellow
        if ($missingModels -gt 0) {
            Write-Host "Put the files there, or point `$env:ANIMA_MODEL_ROOT or examples\anima\models.local.ps1 at a folder that has them (see README.md)." -ForegroundColor Yellow
        }
        exit 1
    }

    $outputBase = Join-Path $AnimaRepoRoot "output\ANIMA"
    $trainerArgs = @(
        "--config_file", $config,
        "--dataset_config", $datasetConfig,
        "--output_dir", (Join-Path $outputBase $outputName),
        "--logging_dir", (Join-Path $outputBase ("log\" + $outputName)),
        "--output_name=$outputName",
        "--vae_chunk_size=64",
        "--vae_disable_cache"
    ) + $setup.ModelArgs

    Write-Host "model:   $($setup.Files[0].Path)" -ForegroundColor Green
    Write-Host "config:  $config" -ForegroundColor Green
    Write-Host "dataset: $datasetConfig" -ForegroundColor Green
    Write-Host "output:  $(Join-Path $outputBase $outputName)" -ForegroundColor Green
    if ($extraArgs.Count -gt 0) { Write-Host "extra:   $($extraArgs -join ' ')" -ForegroundColor Green }

    # run from the repo root: relative paths in the configs and dataset TOMLs are resolved from there
    $exitCode = 1
    Push-Location -LiteralPath $AnimaRepoRoot
    try {
        & $activate
        & $accelerateExe launch `
            --dynamo_backend no `
            --dynamo_mode default `
            --mixed_precision bf16 `
            --num_processes 1 `
            --num_machines 1 `
            --num_cpu_threads_per_process $setup.Threads `
            $trainer `
            @trainerArgs `
            @extraArgs
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) { $exitCode = 1 }
    } finally {
        Pop-Location
    }
    exit $exitCode
}
