# tmac-seedvc

A small command line wrapper that runs [Seed-VC](https://github.com/Plachtaa/seed-vc) (zero-shot voice conversion) on one source recording and
one short reference voice, and writes the converted speech as a WAV file. It is licensed under the **GNU General Public License v3.0 or later**
(see `LICENSE`), the same family of license as Seed-VC (GPL-3.0).

TM AutoClip can use Seed-VC for voice conversion. Because Seed-VC is GPL-3.0, it is **not** built into TM AutoClip: TM AutoClip downloads this
wrapper separately and starts it as a separate process. The two programs talk only through a request file, audio files and standard error lines.
This wrapper does not contain or import any TM AutoClip code.

> **Maintenance:** the Seed-VC repository was archived by its author (read-only). The wrapper is pinned to its last commit and does not follow
> later changes.

## What it needs

- Python 3.10 or newer with `torch`, `torchaudio`, `librosa`, `transformers`, `pyyaml`, `munch`, `einops`, `descript-audio-codec`, `matplotlib`,
  `scipy` and `huggingface_hub` (the versions Seed-VC's own `requirements.txt` lists work). Instead of installing `descript-audio-codec`, its
  two files Seed-VC uses (`dac/nn/quantize.py`, `dac/nn/layers.py`, MIT) can sit in the code folder as a `dac` package.
- Seed-VC's model code: the `modules/` folder and `campplus_cn_common.bin` from Seed-VC, commit `51383efd921027683c89e5348211d93ff12ac2a8`
  (GPL-3.0), in one folder (`paths.code`).
- One Seed-VC checkpoint and its config from <https://huggingface.co/Plachta/Seed-VC> (GPL-3.0), revision `257283f9f41585055e8f858fba4fd044e5caed6e`:
  - speech, 22 kHz: `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` + `config_dit_mel_seed_uvit_whisper_small_wavenet.yml`
  - singing, 44 kHz (pitch follows the source): `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` +
    `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`
- The models Seed-VC loads next to its checkpoint, each in its own folder:
  - content encoder: <https://huggingface.co/openai/whisper-small> (Apache-2.0)
  - vocoder: <https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x> (speech) or <https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x>
    (singing) (MIT)
  - pitch extractor for singing: `rmvpe.pt` from <https://huggingface.co/lj1995/VoiceConversionWebUI> (MIT)

No file is downloaded by the wrapper (it sets `HF_HUB_OFFLINE=1`).

## Usage

```
python tmac_seedvc.py --request request.json
```

```json
{
  "engine": "seed-vc-v1",
  "model": "C:/path/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
  "output": "C:/path/converted.wav",
  "device": "cuda",
  "params": {
    "source": "C:/path/source.wav",
    "voice_sample": "C:/path/reference.wav",
    "seed": 0,
    "diffusion_steps": 30,
    "inference_cfg_rate": 0.7,
    "semitones": 0
  },
  "paths": {
    "code": "C:/path/seed-vc-code",
    "config": "C:/path/config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    "whisper": "C:/path/whisper-small",
    "vocoder": "C:/path/bigvgan_v2_22khz_80band_256x",
    "rmvpe": null
  }
}
```

- `engine`: `seed-vc-v1` (speech, 22 kHz) or `seed-vc-v1-f0` (singing, 44 kHz; needs `paths.rmvpe`; `semitones` shifts the pitch).
- Only the first 25 seconds of the reference voice are used (as in Seed-VC's `inference.py`); long sources are converted in windows with a short
  crossfade.
- Progress lines on standard error: `TMAC_PROGRESS <0-100>`. Errors: `TMAC_ERROR <code> <message>` and a non-zero exit code.
- Seed-VC's checkpoints contain plain Python objects next to the tensors, so they are loaded with `weights_only=False`; only pass checkpoints
  whose checksums you have verified.

## Source and license

- This wrapper: GPL-3.0-or-later, this repository. The conversion steps follow Seed-VC's `inference.py`.
- Seed-VC: GPL-3.0, <https://github.com/Plachtaa/seed-vc> (Songting Liu, "Zero-shot Voice Conversion with Diffusion Transformers", 2024).
- Whisper small: Apache-2.0 (OpenAI). BigVGAN v2: MIT (NVIDIA). RMVPE weights: MIT (RVC project).
