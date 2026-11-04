# tmac-seedvc: a small command line wrapper that runs Seed-VC (zero-shot voice conversion) on one source recording and one reference voice.
#
# Copyright (C) 2026 TM AutoClip
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# The conversion follows Seed-VC's own inference.py (github.com/Plachtaa/seed-vc, commit 51383efd921027683c89e5348211d93ff12ac2a8), with every model
# loaded from local files given in the request (no network access).
import argparse
import functools
import json
import os
import random
import sys
import traceback


def progress(percent):
    print(f"TMAC_PROGRESS {int(max(0, min(100, percent)))}", file=sys.stderr, flush=True)


def fail(code, message):
    print(f"TMAC_ERROR {code} {message}", file=sys.stderr, flush=True)
    sys.exit(2)


def need(mapping, key, kind="file"):
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        fail("request", f"missing '{key}'")
    exists = os.path.isdir(value) if kind == "dir" else os.path.isfile(value)
    if not exists:
        fail("request", f"'{key}' does not exist")
    return value


def crossfade(chunk1, chunk2, overlap, np):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def main():
    parser = argparse.ArgumentParser(description="Seed-VC voice conversion (GPL-3.0-or-later wrapper)")
    parser.add_argument("--request", required=True)
    request = json.load(open(parser.parse_args().request, encoding="utf-8"))
    engine = request.get("engine")
    if engine in ("seed-vc-v2", "seed-vc-v2-style"):
        run_v2(request, convert_style=engine == "seed-vc-v2-style")
        return
    if engine not in ("seed-vc-v1", "seed-vc-v1-f0"):
        fail("request", f"unknown engine '{engine}'")
    singing = engine == "seed-vc-v1-f0"
    params = request.get("params") or {}
    paths = request.get("paths") or {}
    checkpoint = need(request, "model")
    output = request.get("output")
    if not isinstance(output, str) or not output:
        fail("request", "missing 'output'")
    source = need(params, "source")
    target = need(params, "voice_sample")
    code = need(paths, "code", "dir")
    config_path = need(paths, "config")
    whisper_dir = need(paths, "whisper", "dir")
    vocoder_dir = need(paths, "vocoder", "dir")
    rmvpe_path = need(paths, "rmvpe") if singing else None
    campplus_path = os.path.join(code, "campplus_cn_common.bin")
    if not os.path.isfile(campplus_path):
        fail("request", "campplus_cn_common.bin is missing from the Seed-VC code folder")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, code)
    progress(2)

    import numpy as np
    import torch
    import torchaudio
    import librosa
    import yaml

    # Checkpoints are verified by SHA-256 before they reach this folder; Seed-VC's checkpoints hold plain Python objects next to the tensors.
    torch.load = functools.partial(torch.load, weights_only=False)

    seed = int(params.get("seed", 0))
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if request.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    fp16 = device.type == "cuda"
    diffusion_steps = int(params.get("diffusion_steps", 30))
    cfg_rate = float(params.get("inference_cfg_rate", 0.7))
    semitones = int(params.get("semitones", 0))

    from modules.commons import recursive_munch, build_model, load_checkpoint
    from modules.campplus.DTDNN import CAMPPlus
    from modules.bigvgan import bigvgan
    from modules.audio import mel_spectrogram
    from transformers import AutoFeatureExtractor, WhisperModel

    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    model, _, _, _ = load_checkpoint(model, None, checkpoint, load_only_params=True, ignore_modules=[], is_distributed=False)
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    progress(15)

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(campplus_path, map_location="cpu"))
    campplus.eval().to(device)

    vocoder = bigvgan.BigVGAN.from_pretrained(vocoder_dir, use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    whisper = WhisperModel.from_pretrained(whisper_dir, torch_dtype=torch.float16 if fp16 else torch.float32).to(device)
    del whisper.decoder
    extractor = AutoFeatureExtractor.from_pretrained(whisper_dir)
    progress(30)

    def semantic(waves_16k):
        inputs = extractor([waves_16k.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True)
        features = whisper._mask_input_features(inputs.input_features, attention_mask=inputs.attention_mask).to(device)
        with torch.no_grad():
            out = whisper.encoder(features.to(whisper.encoder.dtype), head_mask=None, output_attentions=False, output_hidden_states=False,
                                  return_dict=True)
        hidden = out.last_hidden_state.to(torch.float32)
        return hidden[:, :waves_16k.size(-1) // 320 + 1]

    f0_fn = None
    if singing:
        from modules.rmvpe import RMVPE
        f0_fn = RMVPE(rmvpe_path, is_half=False, device=device).infer_from_audio

    spect = config["preprocess_params"]["spect_params"]
    sr = config["preprocess_params"]["sr"]
    mel_args = {
        "n_fft": spect["n_fft"], "win_size": spect["win_length"], "hop_size": spect["hop_length"], "num_mels": spect["n_mels"],
        "sampling_rate": sr, "fmin": spect.get("fmin", 0), "fmax": None if spect.get("fmax", "None") == "None" else 8000, "center": False,
    }
    hop = spect["hop_length"]
    max_context = sr // hop * 30
    overlap_frames = 16
    overlap_wave = overlap_frames * hop

    with torch.no_grad():
        source_audio = torch.tensor(librosa.load(source, sr=sr)[0]).unsqueeze(0).float().to(device)
        ref_audio = torch.tensor(librosa.load(target, sr=sr)[0][:sr * 25]).unsqueeze(0).float().to(device)
        source_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        if source_16k.size(-1) <= 16000 * 30:
            s_alt = semantic(source_16k)
        else:
            overlap_s = 5
            parts, buffer, traversed = [], None, 0
            while traversed < source_16k.size(-1):
                chunk = source_16k[:, traversed:traversed + 16000 * 30] if buffer is None else torch.cat(
                    [buffer, source_16k[:, traversed:traversed + 16000 * (30 - overlap_s)]], dim=-1)
                part = semantic(chunk)
                parts.append(part if traversed == 0 else part[:, 50 * overlap_s:])
                buffer = chunk[:, -16000 * overlap_s:]
                traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlap_s
            s_alt = torch.cat(parts, dim=1)
        ref_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        s_ori = semantic(ref_16k)
        mel = mel_spectrogram(source_audio.float(), **mel_args)
        mel2 = mel_spectrogram(ref_audio.float(), **mel_args)
        target_lengths = torch.LongTensor([mel.size(2)]).to(device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(device)
        feat2 = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus(feat2.unsqueeze(0))
        progress(45)

        f0_ori = shifted = None
        if singing:
            f0_ori = torch.from_numpy(f0_fn(ref_16k[0], thred=0.03)).to(device)[None]
            f0_alt = torch.from_numpy(f0_fn(source_16k[0], thred=0.03)).to(device)[None]
            log_alt = torch.log(f0_alt + 1e-5)
            voiced_ori = torch.log(f0_ori[f0_ori > 1] + 1e-5)
            voiced_alt = torch.log(f0_alt[f0_alt > 1] + 1e-5)
            shifted_log = log_alt.clone()
            if voiced_ori.numel() > 0 and voiced_alt.numel() > 0:
                shifted_log[f0_alt > 1] = log_alt[f0_alt > 1] - torch.median(voiced_alt) + torch.median(voiced_ori)
            shifted = torch.exp(shifted_log)
            if semitones != 0:
                shifted[f0_alt > 1] = shifted[f0_alt > 1] * (2 ** (semitones / 12))

        cond, *_ = model.length_regulator(s_alt, ylens=target_lengths, n_quantizers=3, f0=shifted)
        prompt, *_ = model.length_regulator(s_ori, ylens=target2_lengths, n_quantizers=3, f0=f0_ori)
        window = max_context - mel2.size(2)
        done, chunks, previous = 0, [], None
        total = max(1, cond.size(1))
        while done < cond.size(1):
            chunk_cond = cond[:, done:done + window]
            last = done + window >= cond.size(1)
            cat = torch.cat([prompt, chunk_cond], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
                vc = model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(device), mel2, style2, None, diffusion_steps,
                                         inference_cfg_rate=cfg_rate)
                vc = vc[:, :, mel2.size(-1):]
            wave = vocoder(vc.float()).squeeze()[None, :]
            if done == 0:
                if last:
                    chunks.append(wave[0].cpu().numpy())
                    break
                chunks.append(wave[0, :-overlap_wave].cpu().numpy())
                previous = wave[0, -overlap_wave:]
                done += vc.size(2) - overlap_frames
            elif last:
                chunks.append(crossfade(previous.cpu().numpy(), wave[0].cpu().numpy(), overlap_wave, np))
                break
            else:
                chunks.append(crossfade(previous.cpu().numpy(), wave[0, :-overlap_wave].cpu().numpy(), overlap_wave, np))
                previous = wave[0, -overlap_wave:]
                done += vc.size(2) - overlap_frames
            progress(45 + 50 * min(1.0, done / total))

        result = torch.tensor(np.concatenate(chunks))[None, :].float()
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        torchaudio.save(output, result.cpu(), sr)
    progress(100)


def resolve_target(path):
    """Imports 'package.module.Name.attr' (the longest importable module prefix, then attributes)."""
    import importlib
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            target = getattr(target, name)
        return target
    raise ImportError(f"cannot import '{path}'")


def instantiate(node):
    """A small stand-in for hydra.utils.instantiate (Seed-VC v2's config uses _target_ and _partial_ only)."""
    if isinstance(node, list):
        return [instantiate(item) for item in node]
    if not isinstance(node, dict):
        return node
    if "_target_" not in node:
        return {key: instantiate(value) for key, value in node.items()}
    target = resolve_target(node["_target_"])
    partial = bool(node.get("_partial_", False))
    kwargs = {key: instantiate(value) for key, value in node.items() if key not in ("_target_", "_partial_")}
    return functools.partial(target, **kwargs) if partial else target(**kwargs)


def run_v2(request, convert_style):
    """Seed-VC v2 (content: ASTRAL quantizers over HuBERT, timbre: CFM, accent and emotion: optional AR model), following modules/v2/vc_wrapper.py."""
    params = request.get("params") or {}
    paths = request.get("paths") or {}
    cfm_path = need(request, "model")
    output = request.get("output")
    if not isinstance(output, str) or not output:
        fail("request", "missing 'output'")
    source = need(params, "source")
    target = need(params, "voice_sample")
    code = need(paths, "code", "dir")
    ar_path = need(paths, "ar")
    narrow_path = need(paths, "astral_narrow")
    wide_path = need(paths, "astral_wide")
    hubert_dir = need(paths, "hubert", "dir")
    tokenizer_dir = need(paths, "tokenizer", "dir")
    vocoder_dir = need(paths, "vocoder", "dir")
    config_path = os.path.join(code, "configs", "v2", "vc_wrapper.yaml")
    campplus_path = os.path.join(code, "campplus_cn_common.bin")
    for required in (config_path, campplus_path):
        if not os.path.isfile(required):
            fail("request", f"{os.path.basename(required)} is missing from the Seed-VC code folder")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, code)
    # vc_wrapper.py imports pydub only to stream MP3 previews; this wrapper never streams, so a placeholder module is enough.
    import types
    pydub = types.ModuleType("pydub")

    def no_mp3(*_args, **_kwargs):
        raise RuntimeError("MP3 streaming is not used by tmac-seedvc")

    pydub.AudioSegment = no_mp3
    sys.modules.setdefault("pydub", pydub)
    progress(2)

    import numpy as np
    import random as std_random
    import torch
    import soundfile
    import yaml

    torch.load = functools.partial(torch.load, weights_only=False)
    seed = int(params.get("seed", 0))
    if seed >= 0:
        std_random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if request.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    for key in ("content_extractor_narrow", "content_extractor_wide"):
        config[key]["tokenizer_name"] = tokenizer_dir
        config[key]["ssl_model_name"] = hubert_dir
    config["vocoder"]["pretrained_model_name_or_path"] = vocoder_dir
    config["vocoder"]["use_cuda_kernel"] = False
    wrapper = instantiate(config)
    progress(15)

    strip = wrapper.strip_prefix
    cfm = torch.load(cfm_path, map_location="cpu")
    wrapper.cfm.load_state_dict(strip(cfm["net"]["cfm"], "module."), strict=False)
    wrapper.cfm_length_regulator.load_state_dict(strip(cfm["net"]["length_regulator"], "module."), strict=False)
    ar = torch.load(ar_path, map_location="cpu")
    wrapper.ar.load_state_dict(strip(ar["net"]["ar"], "module."), strict=False)
    wrapper.ar_length_regulator.load_state_dict(strip(ar["net"]["length_regulator"], "module."), strict=False)
    wrapper.content_extractor_narrow.load_state_dict(torch.load(narrow_path, map_location="cpu"), strict=False)
    wrapper.content_extractor_wide.load_state_dict(torch.load(wide_path, map_location="cpu"), strict=False)
    wrapper.style_encoder.load_state_dict(torch.load(campplus_path, map_location="cpu"), strict=False)
    wrapper.to(device)
    wrapper.eval()
    wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    progress(30)

    # The upstream generator collects the whole result only while streaming MP3; the chunk list it passes to _stream_wave_chunks is the result.
    captured = {}
    original = type(wrapper)._stream_wave_chunks

    def collect(self, vc_wave, processed_frames, vc_mel, overlap_wave_len, generated_wave_chunks, previous_chunk, is_last_chunk, _stream):
        captured["chunks"] = generated_wave_chunks
        result = original(self, vc_wave, processed_frames, vc_mel, overlap_wave_len, generated_wave_chunks, previous_chunk, is_last_chunk, False)
        progress(30 + 65 * (1.0 if is_last_chunk else min(0.95, len(generated_wave_chunks) / 20.0)))
        return result

    type(wrapper)._stream_wave_chunks = collect
    with torch.no_grad():
        for _ in wrapper.convert_voice_with_streaming(
                source_audio_path=source,
                target_audio_path=target,
                diffusion_steps=int(params.get("diffusion_steps", 30)),
                length_adjust=float(params.get("length_adjust", 1.0)),
                intelligebility_cfg_rate=float(params.get("intelligibility_cfg_rate", 0.7)),
                similarity_cfg_rate=float(params.get("similarity_cfg_rate", 0.7)),
                top_p=float(params.get("top_p", 0.9)),
                temperature=float(params.get("temperature", 1.0)),
                repetition_penalty=float(params.get("repetition_penalty", 1.0)),
                convert_style=convert_style,
                anonymization_only=False,
                device=device,
                dtype=dtype,
                stream_output=False):
            pass
    chunks = captured.get("chunks")
    if not chunks:
        fail("runtime", "Seed-VC v2 produced no audio")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    soundfile.write(output, np.concatenate(chunks), wrapper.sr)
    progress(100)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - the caller needs the reason as a single line
        traceback.print_exc(file=sys.stderr)
        fail("runtime", f"{type(error).__name__}: {error}".replace("\n", " ")[:400])
