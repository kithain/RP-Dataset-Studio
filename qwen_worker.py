"""Private JSONL worker using the user's ComfyUI Python runtime and local weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-dir", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, str(Path(args.comfy_dir).resolve()))
    # ComfyUI's argument parser must not consume the worker's arguments.
    sys.argv = [sys.argv[0]]
    import comfy.options
    comfy.options.enable_args_parsing(False)
    from comfy.cli_args import args as comfy_args
    comfy_args.disable_dynamic_vram = True
    import torch
    import comfy.sd

    clip = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if clip is None:
                clip = comfy.sd.load_clip(ckpt_paths=[str(Path(args.model).resolve())],
                                          disable_dynamic=True)
            if request.get("op") == "load":
                result = {"device": str(clip.patcher.load_device), "model": Path(args.model).name}
            elif request.get("op") == "generate":
                tokens = clip.tokenize(request["prompt"], system_prompt=request["system"],
                                       thinking=False, min_length=1)
                count = sum(len(batch) for batches in tokens.values() for batch in batches)
                if count > 8192:
                    raise ValueError(f"Contexte trop long : {count} tokens (maximum 8192).")
                with torch.inference_mode():
                    ids = clip.generate(tokens, do_sample=False, max_length=512,
                                        repetition_penalty=1.05, seed=42, mtp=False)
                result = {"text": clip.decode(ids), "input_tokens": count}
            else:
                raise ValueError("Opération inconnue")
            response = {"ok": True, **result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": str(exc)}
        protocol.write(json.dumps(response, ensure_ascii=True) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
