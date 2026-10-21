# Anima LoRA / full fine-tune examples

Ready-to-run PowerShell launchers and configs for training on the Anima models with `backend\sd_scripts`.
LoRAs use LyCORIS LoKr + DoRA. After `install.bat`, you only need the model files and a dataset.
The launchers are for Windows (Windows PowerShell 5.1 or PowerShell 7; they use the `venv\Scripts` layout).
On Linux, call `accelerate launch` yourself with the same arguments the launcher prints and builds.

## Files

| File | What it is |
|---|---|
| `train_anima_base.ps1` | LoRA on **Anima base 1.0** (28 blocks), uses `config_anima_base.toml` |
| `train_anima29.ps1` | LoRA on **Anima-2.9B preview v1** (40 blocks), uses `config_anima29.toml` |
| `train_anima38.ps1` | LoRA on **Anima-3.8B v1.1** (52 blocks, Qwen3.5-4B + Semantic Connector v2), uses `config_anima38.toml` |
| `train_anima_fft.ps1` | **Experimental:** full fine-tune of **Anima base 1.0** (`anima_train.py`), uses `config_anima_fft.toml`. Unlike the LoRA recipes, it is not yet validated end to end and its hyperparameters are not final. |
| `models.ps1` | The one place that says where the model files are. All launchers read it. Do not edit it; use `models.local.ps1` (below). |
| `anima_common.ps1` | Shared launcher code: checks the files, then starts `accelerate`. You do not run it directly. |
| `config_anima_*.toml` | Training settings. The three LoRA configs share one recipe and differ only in the model, activation offload, and the 3.8B connector block. |
| `dataset_example.toml` | Dataset template: four resolution tiers (2048 / 1536 / 1024 / 512), caption shuffle with `\|\|\|`. |
| `sample_anima.txt` | Prompts for the sample images rendered during training (with an `@your_trigger` placeholder; see section 4). |

The tracked files here are meant to be used as they are: `update.bat` runs `git pull`, which stops if you edited a
tracked file that also changed upstream. Put your own settings in copies or on the command line, as shown below.

## 1. Install

Install with the `git clone` command in the main README, then run `install.bat` in the repo root and answer `y` to
*Are you using this locally?*. This creates `backend\sd_scripts\venv` with PyTorch, LyCORIS and flash-attn
(Windows, Python 3.11). A GitHub ZIP download has no submodules, so `install.bat` cannot set up the trainer from it.

The configs use `attn_mode = "flash"`. If flash-attn is not available on your system, add `--attn_mode torch` to the
launcher command (see section 4) instead of editing the config.

## 2. Download the model files

Put them under `models\` in the repo root (Forge-style subfolders; `models\` is git-ignored).
Each launcher only needs the files for its model. No login is needed for any of them.

| File | Goes to | Needed by | Download (Hugging Face) |
|---|---|---|---|
| `anima-base-v1.0.safetensors` | `models\Stable-diffusion\` | base, fft | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-base-v1.0.safetensors) `split_files/diffusion_models/` |
| `Anima-2.9B-preview-v1.safetensors` | `models\Stable-diffusion\` | 2.9B | [Gazingstars123/Anima-2.9B](https://huggingface.co/Gazingstars123/Anima-2.9B/blob/main/Anima-2.9B-preview-v1.safetensors) (not the int8 file) |
| `Anima-3.8B-v1.1.safetensors` | `models\Stable-diffusion\` | 3.8B | [lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B/blob/main/difussion_models/Anima-3.8B-v1.1.safetensors) folder `difussion_models/` (spelled that way in the repo). The connector is inside this file. |
| `qwen_3_06b_base.safetensors` | `models\text_encoder\` | all | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/text_encoders/qwen_3_06b_base.safetensors) `split_files/text_encoders/` |
| `qwen35_4b.safetensors` | `models\text_encoder\` | 3.8B | [lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B/blob/main/text_encoders/qwen35_4b.safetensors) `text_encoders/` |
| `qwen_image_vae.safetensors` | `models\VAE\` | all | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/vae/qwen_image_vae.safetensors) `split_files/vae/` |
| `tokenizer.json`, `tokenizer_config.json` | `models\qwen35_tokenizer\` | 3.8B | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) (only these two files) |

```
models\
  Stable-diffusion\  anima-base-v1.0.safetensors, Anima-2.9B-preview-v1.safetensors, Anima-3.8B-v1.1.safetensors
  text_encoder\      qwen_3_06b_base.safetensors, qwen35_4b.safetensors
  VAE\               qwen_image_vae.safetensors
  qwen35_tokenizer\  tokenizer.json, tokenizer_config.json
```

The Qwen3-0.6B and T5 tokenizers ship with the trainer (`backend\sd_scripts\configs\`); nothing is downloaded at run time.
Check each model's license on its page before you use or share what you train: Anima base and Anima-2.9B are under
the circlestone-labs non-commercial license (for 2.9B see `LICENSE.md` in its repo; there is nothing to accept before
downloading), Anima-3.8B is built on them, and Qwen3.5-4B is Apache-2.0.

### Already have them in Forge / ComfyUI?

No need to copy. Set the folder for the current PowerShell window:

```powershell
$env:ANIMA_MODEL_ROOT = "C:\path\to\forge\models"          # has Stable-diffusion\, text_encoder\, VAE\
$env:ANIMA38_QWEN35_TOKENIZER_DIR = "C:\path\to\qwen35_tokenizer"   # 3.8B only, if it is not <root>\qwen35_tokenizer
```

or, to keep it, create `examples\anima\models.local.ps1` (git-ignored, so `git pull` never touches it).
`models.ps1` reads it last, so any variable it sets wins:

```powershell
# examples\anima\models.local.ps1 (UTF-8, with or without BOM)
$AnimaModelRoot = "C:\path\to\forge\models"
$AnimaQwen35TokenizerDir = "C:\path\to\qwen35_tokenizer"
```

If your files have other names (for example a Civitai download), set the file name variables (`$AnimaBaseFile`,
`$Anima29File`, `$Anima38File`, `$AnimaQwen3File`, `$AnimaQwen35File`, `$AnimaVaeFile`) in `models.local.ps1`.
For ComfyUI, whose folders are named differently, set the folders there too:

```powershell
$AnimaDitDir = "C:\path\to\ComfyUI\models\diffusion_models"
$AnimaTextEncoderDir = "C:\path\to\ComfyUI\models\text_encoders"
$AnimaVaeDir = "C:\path\to\ComfyUI\models\vae"
$Anima38File = "Anima-3.8B-v2.safetensors"   # only if you renamed the 3.8B bundle as the ComfyUI instructions suggest
```

## 3. Make a dataset

1. Put images and same-named caption files in a folder, e.g. `datasets\my_style\best\001.png` + `001.txt`
   (`datasets\` is git-ignored). Caption: `@your_trigger ||| 1girl, solo, ...` (tags before `|||` stay first; the rest is shuffled).
2. Copy `dataset_example.toml` (e.g. to `datasets\my_style.toml`) and edit `image_dir` and `num_repeats`.
   Relative `image_dir` paths are resolved from the repo root.

Save the dataset TOML, the captions and prompt files as **UTF-8 without BOM** (Windows 11 Notepad "UTF-8" is fine;
avoid "UTF-8 with BOM" and PowerShell 5.1 `Out-File` / `Set-Content -Encoding UTF8`). The launcher refuses a dataset
TOML with a BOM; in a caption a BOM would silently become part of the trigger.

The caches (`*_anima.npz` latents, `*_anima_te.npz` / `*_anima38_te.npz` text features) are written next to the images.

## 4. Train

Run from the repo root in PowerShell:

```powershell
.\examples\anima\train_anima_base.ps1 .\datasets\my_style.toml my_style_base
.\examples\anima\train_anima29.ps1   .\datasets\my_style.toml my_style_29
.\examples\anima\train_anima38.ps1   .\datasets\my_style.toml my_style_38
.\examples\anima\train_anima_fft.ps1 .\datasets\my_style.toml my_anima_finetune   # experimental
```

- Results: `output\ANIMA\<output_name>\`, logs: `output\ANIMA\log\<output_name>\`, sample images in `output\ANIMA\<output_name>\sample\`
  (`output\` is git-ignored).
- Any extra arguments go to the trainer and override the config, e.g.
  `.\examples\anima\train_anima29.ps1 .\datasets\my_style.toml my_style_29 --max_train_epochs=10 --learning_rate 1e-5`.
- Your own sample prompts: copy `sample_anima.txt` (e.g. to `datasets\my_samples.txt`), replace `@your_trigger`, and add
  `--sample_prompts=.\datasets\my_samples.txt`. No flash-attn: add `--attn_mode torch`.
- PowerShell quoting: quote a name that starts with `@` (`'@my_lora'`; unquoted, PowerShell treats it as splatting and
  the launcher only prints its usage line) and any single value that contains a comma (`'a,b'`); an unquoted `a,b`
  is passed as two arguments `a` `b`. List options take one argument per item: `--network_args 'algo=lokr' 'factor=8'`;
  a list on the command line replaces the config's whole list, so repeat every item you keep.
- Resume: add `--resume=output\ANIMA\<output_name>\<output_name>-000003-state` (states are saved every epoch).
  Do not end the path with a backslash: with a quoted path that contains a space, Windows PowerShell 5.1 turns a
  trailing `\` into a stray `"`.
- If a file is missing, the launcher stops before anything is loaded, prints the expected path and the download link,
  and exits with code 1.

Recipe (LoRA): LoKr factor 4 full matrix + DoRA, AdamW lr 2e-5 (betas 0.9/0.99, wd 0.01), constant with 50 warmup steps,
gradient accumulation 4, 30 epochs, sigmoid timesteps (scale 1.3), ip_noise_gamma 0.05, bf16, fp16 saves, cached
text-encoder outputs with 16 caption variants. 3.8B also offloads checkpointed activations to CPU RAM.
VRAM depends mostly on the dataset's resolution tiers (`batch_size` is already 1); drop the 2048 tier if you run out.

## "running scripts is disabled on this system"

Windows blocks `.ps1` files by default. Either allow your own scripts once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or run a launcher with a one-time bypass (nothing is changed permanently):

```powershell
powershell -ExecutionPolicy Bypass -File .\examples\anima\train_anima29.ps1 .\datasets\my_style.toml my_style_29
```

If you downloaded single `.ps1` files with a browser (not through `git clone`), unblock them once with `Unblock-File`.
