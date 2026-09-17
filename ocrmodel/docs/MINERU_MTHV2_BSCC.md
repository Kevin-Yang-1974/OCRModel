# MinerU2.5-Pro on BSCC: MTHv2 SFT compatibility

## Scope

This run is a whole-page text-recognition SFT adaptation of the staged
`MinerU2.5-Pro-2605-1.2B` Qwen2-VL checkpoint on the existing MTHv2 split:

- train: 2,159 pages;
- validation: 240 pages;
- test: not opened by preparation or training;
- prompt: `<image>\nText Recognition:`;
- formal budget: four GPUs, 5,000 optimizer steps, LoRA rank 8.

The target is MTHv2 `page_text`; this is not a claim of full MinerU document
parsing supervision for tables, formulas, or layout regions.

## Direct versus compatibility path

The OpenDataLab training tutorial documents the official command shape
`python run_mineru.py sft --config ...` and says that the command loads
`mineru_ext`. The public MinerU2.5-Pro-2605 model card exposes a standard
`Qwen2VLForConditionalGeneration` checkpoint, but the public MinerU repository
does not contain the tutorial's SFT archive.

Therefore BSCC uses the following compatibility boundary:

1. `mineru_mthv2_compat.py` converts MTHv2 to standard MS-SWIFT multimodal
   JSONL and writes an absolute-path YAML configuration.
2. `run_mineru.py` preserves the official entry-point invocation and forwards
   to the official MS-SWIFT `sft` CLI installed in an isolated environment.
3. `run_mineru2_5_pro_mthv2_4gpu.sbatch` performs GPU admission, data/protocol
   preparation, a non-training compatibility preflight, and one formal
   four-process SFT launch.

The resulting run must be described as `bscc_ms_swift_forwarder` until the
original `mineru_ext` archive is available for a byte-for-byte direct run.

## Selection boundary

The training script saves checkpoints and evaluates only the validation split.
Validation-only checkpoint selection and any later selection-locked MTHv2 test
run remain separate actions; no test file is read here.
