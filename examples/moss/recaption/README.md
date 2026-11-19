# MOSS-Image style recaptioning

OpenMOSS published how they caption images for post-training MOSS-Image-Base
([USERGUIDE "Image Captioning"](https://github.com/OpenMOSS/MOSS-Image/blob/c1355e6/USERGUIDE.md#image-captioning),
2026-10-16): a VLM rewrites each caption as one JSON object with `long` / `medium` / `short` descriptions plus
watermark boxes and borders, with the original caption given as a reference. The technical report trains on the
original captions 20% of the time and on the recaptions 80% (the three lengths 1:1:1).

This folder reproduces that for a LoRA dataset with `moss_recaption.py`, using Codex CLI on the ChatGPT plan as the VLM
(one call per image: prompt + image, the JSON shape enforced by `caption_schema.json`).

| file | |
|---|---|
| `prompt_official.txt` | the official prompt, verbatim |
| `prompt_no_style.txt` | the same without describing the art style (see below) |
| `caption_schema.json` | the official JSON shape, for `codex exec --output-schema` |
| `moss_recaption.py` | copy / queue / run / check / build |

## Why two prompts

The official long caption describes the "artistic style". In a style LoRA the trigger (e.g. `@mytrigger`) should carry
the style; if every caption also says "soft painterly anime illustration", the words can take over part of that job,
and prompts without those words may get less of the style. `no_style` keeps everything else and only drops style
descriptions. Caption a pilot with both and compare before captioning the whole set.

## Steps (PowerShell, from the repository root)

```powershell
$py = ".\backend\sd_scripts\venv_t214\Scripts\python.exe"
$kit = ".\examples\moss\recaption\moss_recaption.py"
$work = "path\to\my_dataset_moss_recap"   # a new folder outside the dataset

# 1. copy the dataset (dry run first; the source folders are never written)
& $py $kit copy --dataset-config path\to\my_dataset_config.toml --work $work
& $py $kit copy --dataset-config path\to\my_dataset_config.toml --work $work --apply

# 2. queue every image; pick a 10-image pilot (stratified by subset)
& $py $kit queue --work $work --trigger "@mytrigger" --pilot 10

# 3. pilot with both prompts, then review
& $py $kit run --work $work --queue pilot.jsonl --prompt official
& $py $kit run --work $work --queue pilot.jsonl --prompt no_style
& $py $kit check --work $work --prompt official --trigger "@mytrigger"
& $py $kit check --work $work --prompt no_style --trigger "@mytrigger"

# 4. after the review: the rest with the chosen prompt (images that already have a JSON are skipped)
& $py $kit run --work $work --queue queue.jsonl --prompt no_style

# 5. training captions (dry run first)
& $py $kit build --work $work --prompt no_style --trigger "@mytrigger"
& $py $kit build --work $work --prompt no_style --trigger "@mytrigger" --apply

# 6. train on the copy (16 caption variants: see "Caption mix")
.\examples\moss\train_moss.ps1 "$work\dataset_config_moss_recap.toml" my_lora_recap_v1 --cache_caption_variants 16
```

`copy` also writes `dataset_config_moss_recap.toml` in the work folder: the source config with `image_dir` pointing at
the copy and `enable_wildcard = true`. Keep everything else the same as the run you compare against.

## What the files look like

`run` writes `<image>.moss_<prompt>.json` beside each copied image and appends to `progress_run_<prompt>.jsonl`; a
re-run skips images that already have a JSON (`--redo` captions them again). Invalid replies go to
`<image>.moss_<prompt>.invalid.json` and the progress file.

`build` writes one caption file per image (the originals stay in `_orig/`, the previous files go to
`_backup_captions_<time>/`):

```
@mytrigger, 1girl ||| solo, blonde hair, ... ||| A woman in a white dress sits ...
@mytrigger ||| ||| <short>
@mytrigger ||| ||| <short>
@mytrigger ||| ||| <medium>
@mytrigger ||| ||| <medium>
@mytrigger ||| ||| <long>
@mytrigger ||| ||| <long>
```

- With `enable_wildcard = true` every line is one alternative. Line 1 is also the fixed variant 0 of the cache.
- `@mytrigger ||| ||| text` keeps the trigger first and stops `shuffle_caption` from shuffling the sentence's clauses
  (the trainer turns it into `@mytrigger, text`).
- Newlines inside a caption become spaces, and `{ }` are doubled so they stay literal under `enable_wildcard`.

## Caption mix

Each image gets `cache_caption_variants` cached captions: variant 0 is line 1, the others are random lines, and a
training step uses a random variant. With `--layout 1,2,2,2` (the default) and 16 variants the mix is about 20%
original, 27% each of short / medium / long, which is the official 20 : 80 with 1 : 1 : 1. Measured through the
trainer's own variant code: 19.7 / 27.2 / 26.2 / 26.9%.

Long captions make the text-encoder cache bigger and slower to build (16 variants per image instead of 8).

## Notes

- `run` needs Codex signed in with ChatGPT (`~/.codex/auth.json`, `auth_mode = chatgpt`); it refuses otherwise and
  never passes `OPENAI_API_KEY` on, so it cannot bill the API. One synthetic test image took 12 s at effort `high`.
- The model gets no tools (`-s read-only`, an empty scratch folder as its working directory).
- The work folder is a separate copy so the text-encoder caches of the original run and the recaption run never
  overwrite each other. Do not train two runs on the same folder at once.
- `check` flags empty fields, bad watermark boxes or borders, the trigger or `|||` inside captions, captions that
  mention a watermark, and style words (a quick way to compare the two prompts). It does not look at the images:
  review the pilot against the images before captioning the rest.

## Attribution

`prompt_official.txt` and `caption_schema.json` follow the "Image Captioning" section of OpenMOSS/MOSS-Image
`USERGUIDE.md` (commit c1355e6), licensed under the Apache License 2.0 (`LICENSE-moss-image-Apache-2.0.txt`).
`prompt_no_style.txt` is a modified copy: the long-caption rule no longer asks for the artistic style, and two rules
forbid describing it (changed 2026-10-18).
