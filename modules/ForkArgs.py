"""Fork keys that have their own GUI controls, and moving them out of Extra Args when a TOML is loaded (no widgets).

Older GUI TOMLs carried fork options (use_ramtorch, anima38_connector, qwen35_path, ...) as Extra Args. Extra Args are
flattened last by the backend, so they silently override the dedicated controls. On load, a key of an active section is
moved from extra_args into that section (with the Extra Args value, which is the one the trainer saw); after the widgets
have loaded, ``unconfirmed`` lists the moved keys whose control does not write the same value, and those go back to
Extra Args with a warning.
"""

import copy

from modules.SigmoidScaleInput import parse_sigmoid_scale

COMMON_DIT_KEYS = (
    "use_ramtorch", "scale_resume", "sigmoid_scale", "timestep_sampling", "discrete_flow_shift", "split_attn",
    "blocks_to_swap", "unsloth_offload_checkpointing",
)
MOSS_KEYS = COMMON_DIT_KEYS + (
    "moss_tokenizer_dir", "moss_max_token_length", "moss_logit_mean", "moss_logit_std", "moss_sigma_cutoff",
    "moss_uniform_mix", "moss_lora_exclude_blocks", "moss_sample_shift", "moss_sample_uncond_skip_layers",
    "moss_no_compile", "moss_no_lokr_compile", "moss_stock_lycoris", "moss_no_sample_cache", "attn_mode",
)
ANIMA_KEYS = COMMON_DIT_KEYS + (
    "anima_no_compile", "anima_no_lokr_compile", "anima_stock_lycoris", "anima_no_sample_cache",
)
ANIMA38_KEYS = (
    "anima38_connector", "qwen35_path", "qwen35_tokenizer_dir", "anima38_sample_negative", "anima38_qwen35_batch_size",
)
DORA_KEYS = ("learning_rate_DoRA", "weight_decay_DoRA")
CAPTION_VARIANT_KEYS = ("cache_caption_variants",)
ALIASES = {"learning_rate_dora": "learning_rate_DoRA", "weight_decay_dora": "weight_decay_DoRA"}

SECTION_TITLES = {"moss_args": "MOSS-Image Args", "anima_args": "Anima Args", "network_args": "Network Args"}

# Value a control writes no key for (a moved key that comes back absent is confirmed when it equals this).
DEFAULTS = {
    "moss_max_token_length": 4096, "moss_logit_mean": 1.09, "moss_logit_std": 0.95, "moss_sigma_cutoff": 0.95,
    "moss_uniform_mix": 0.01, "sigmoid_scale": 1.0, "scale_resume": 0, "discrete_flow_shift": 1.0,
    "blocks_to_swap": 0, "moss_sample_shift": 5.0, "moss_sample_uncond_skip_layers": "22",
    "moss_lora_exclude_blocks": "", "moss_tokenizer_dir": "", "qwen35_tokenizer_dir": "",
    "anima38_sample_negative": "connector", "anima38_qwen35_batch_size": 1, "attn_mode": None,
    "weight_decay_DoRA": 0.0, "cache_caption_variants": 0,
}
BLOCK_LIST_KEYS = ("moss_lora_exclude_blocks", "moss_sample_uncond_skip_layers")
ATTN_ALIASES = {"sdpa": "torch"}


def canonical(key: str) -> str:
    return ALIASES.get(key, key)


def _network_section(args: dict) -> dict:
    section = args.get("network_args")
    return section if isinstance(section, dict) else {}


def _dora_on(network: dict) -> bool:
    inner = network.get("network_args")
    if isinstance(inner, dict):
        return str(inner.get("dora_wd", "")).strip().lower() == "true"
    if isinstance(inner, list):
        return any(str(item).replace(" ", "").lower() == "dora_wd=true" for item in inner)
    return False


def control_keys(moss_on: bool, anima_on: bool, anima38_on: bool, dora_on: bool, cache_te_on: bool) -> dict:
    """{key: section} for the fork keys a dedicated control writes in this state."""
    keys = {}
    if moss_on:
        keys.update({key: "moss_args" for key in MOSS_KEYS})
    if anima_on:
        keys.update({key: "anima_args" for key in ANIMA_KEYS})
        if anima38_on:
            keys.update({key: "anima_args" for key in ANIMA38_KEYS})
    if dora_on:
        keys.update({key: "network_args" for key in DORA_KEYS})
    if cache_te_on:
        keys.update({key: "network_args" for key in CAPTION_VARIANT_KEYS})
    return keys


def migrate_extra_args(args: dict) -> tuple[dict, list[tuple[str, str, object]]]:
    """Move fork keys of active sections from args["extra_args"] into those sections.
    Returns (new args, [(key, section, value), ...] moved)."""
    args = copy.deepcopy(args)
    extra = args.get("extra_args")
    if not isinstance(extra, dict) or not extra:
        return args, []
    moss = args.get("moss_args") if isinstance(args.get("moss_args"), dict) else {}
    anima = args.get("anima_args") if isinstance(args.get("anima_args"), dict) else {}
    network = _network_section(args)
    anima38_on = bool(anima) and bool(anima.get("anima38_connector") or extra.get("anima38_connector"))
    keys = control_keys(bool(moss), bool(anima), anima38_on, _dora_on(network),
                        bool(network.get("cache_text_encoder_outputs")))
    moved = []
    for name in list(extra):
        key = canonical(name)
        section = keys.get(key)
        if section is None:
            continue
        value = extra.pop(name)
        target = args.setdefault(section, {})
        for alias, canon in ALIASES.items():  # an alias in the section must not shadow the moved value
            if canon == key:
                target.pop(alias, None)
        target[key] = value
        moved.append((key, section, value))
    if not extra:
        del args["extra_args"]
    return args, moved


def _normalise(key: str, value):
    if key in BLOCK_LIST_KEYS:
        from main_ui_files.MossUI import parse_block_list  # imported late: pulls in Qt widgets
        return parse_block_list(value)
    if key == "attn_mode" and isinstance(value, str):
        return ATTN_ALIASES.get(value.strip().lower(), value.strip().lower())
    if key == "sigmoid_scale" and not isinstance(value, bool):
        parsed = parse_sigmoid_scale(value)
        return value if parsed is None else parsed
    if isinstance(value, str):
        return value.strip()
    return value


def same_value(key: str, wanted, written) -> bool:
    wanted, written = _normalise(key, wanted), _normalise(key, written)
    if isinstance(wanted, bool) or isinstance(written, bool):
        return bool(wanted) == bool(written) if isinstance(wanted, bool) and isinstance(written, bool) else False
    if isinstance(wanted, (int, float)) and isinstance(written, (int, float)):
        return abs(float(wanted) - float(written)) <= 1e-9 * max(1.0, abs(float(wanted)))
    return wanted == written


def unconfirmed(moved: list, final_args: dict) -> list[tuple[str, str, object]]:
    """The moved keys whose control does not write the same value (the trainer would get something else)."""
    lost = []
    for key, section, value in moved:
        written = (final_args.get(section) or {})
        if key in written:
            ok = same_value(key, value, written[key])
        elif isinstance(value, bool):
            ok = value is False
        elif key in DEFAULTS:
            default = DEFAULTS[key]
            ok = (value in ("", None) and default in ("", None)) or (
                default not in ("", None) and same_value(key, value, default))
        else:
            ok = value in ("", None)
        if not ok:
            lost.append((key, section, value))
    return lost
