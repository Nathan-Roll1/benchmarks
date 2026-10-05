#!/usr/bin/env python3
"""
Prepare the AMI Meeting Corpus for the diarization and timestamp benchmarks.

Downloads (CC BY 4.0, no registration):
  - AMI manual annotations v1.6.2 (segments + words XML)
  - Per-meeting Mix-Headset audio (all headset channels summed; matches what a
    user would upload, unlike trivially-easy per-speaker IHM channels)

Writes:
    manifests/diarization/diarization_ami.jsonl   (speaker segments)
    manifests/timestamps/timestamps_ami.jsonl     (word start/end ground truth)
    audio under data/benchmarks/{diarization,timestamps}/audio/

The AMI words.xml carries per-word starttime + endtime, which is the reference
for the timestamp benchmark.

Usage:
    python -m benchmarks.datasets.prepare_ami
    python -m benchmarks.datasets.prepare_ami --meetings 5
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .. import config
from . import _util

# Standard AMI scenario (SC) 20-meeting eval partition.
AMI_SC_TEST_SET = [
    "ES2002a", "ES2002b", "ES2002c", "ES2002d",
    "ES2003a", "ES2003b", "ES2003c", "ES2003d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "ES2005a", "ES2005b", "ES2005c", "ES2005d",
    "IS1008a", "IS1008b", "IS1008c", "IS1008d",
]
ANNOTATIONS_ZIP_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)
AUDIO_URL_TEMPLATE = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
)
NITE_NS = "{http://nite.sourceforge.net/}"

# AMI-SDM: the standard, SOTA-comparable condition (single distant mic Array1-01,
# official "only_words" RTTM references from the BUT AMI diarization setup).
# This is the AMI full-corpus-ASR test partition used by diarization_ami_sdm.jsonl.
AMI_SDM_TEST_SET = [
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
    "EN2002a", "EN2002b", "EN2002c", "EN2002d",
]
SDM_AUDIO_URL_TEMPLATE = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Array1-01.wav"
)
SDM_RTTM_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/"
    "only_words/rttms/test/{m}.rttm"
)
# The scored region (UEM) of each meeting; its end time is the manifest duration_s,
# which the diarization benchmark passes to DER as uem_duration.
SDM_UEM_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/"
    "uems/test/{m}.uem"
)


def _download_annotations(ann_dir: Path) -> None:
    ann_dir.mkdir(parents=True, exist_ok=True)
    if any(ann_dir.glob("*.segments.xml")) and any(ann_dir.glob("*.words.xml")):
        return
    zip_path = _util.download(ANNOTATIONS_ZIP_URL, ann_dir / "_ami_manual_1.6.2.zip")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith((".segments.xml", ".words.xml")):
                dest = ann_dir / Path(name).name
                if not dest.exists():
                    dest.write_bytes(zf.read(name))


def _parse_segments(meeting: str, ann_dir: Path) -> list[tuple[float, float, str]]:
    segs: list[tuple[float, float, str]] = []
    for f in sorted(ann_dir.glob(f"{meeting}.*.segments.xml")):
        parts = f.stem.split(".")
        fallback = parts[-2] if len(parts) >= 3 else f.stem
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        for seg in root.iter("segment"):
            start = seg.get("transcriber_start") or seg.get("start")
            end = seg.get("transcriber_end") or seg.get("end")
            who = seg.get("who") or seg.get("speaker") or fallback
            if start and end:
                try:
                    segs.append((float(start), float(end), who))
                except ValueError:
                    pass
    return sorted(segs, key=lambda x: x[0])


def _parse_words(meeting: str, ann_dir: Path) -> list[dict]:
    words: list[dict] = []
    for f in sorted(ann_dir.glob(f"{meeting}.*.words.xml")):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        for w in list(root.iter(f"{NITE_NS}w")) or list(root.iter("w")):
            if w.get("punc") == "true":
                continue
            start, end = w.get("starttime"), w.get("endtime")
            text = (w.text or "").strip()
            if not text or start is None or end is None:
                continue
            try:
                words.append({"word": text, "start": round(float(start), 3),
                              "end": round(float(end), 3)})
            except ValueError:
                continue
    return sorted(words, key=lambda x: x["start"])


def _speaker_transcripts(meeting: str, ann_dir: Path) -> dict[str, str]:
    """Per-speaker reference text (for cpWER). Each words.xml is one speaker;
    the speaker key is the letter in {meeting}.{speaker}.words.xml."""
    out: dict[str, list[tuple[float, str]]] = {}
    for f in sorted(ann_dir.glob(f"{meeting}.*.words.xml")):
        parts = f.stem.split(".")
        spk = parts[-2] if len(parts) >= 3 else f.stem
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        for w in list(root.iter(f"{NITE_NS}w")) or list(root.iter("w")):
            if w.get("punc") == "true":
                continue
            text = (w.text or "").strip()
            start = w.get("starttime")
            if not text or start is None:
                continue
            try:
                out.setdefault(spk, []).append((float(start), text))
            except ValueError:
                continue
    return {spk: " ".join(t for _, t in sorted(ws)) for spk, ws in out.items()}


def prepare(meeting_ids: list[str], download_audio: bool = True) -> None:
    ann_dir = config.DATA_ROOT / "diarization" / "annotations"
    diar_audio = config.audio_dir("diarization")
    ts_audio = config.audio_dir("timestamps")
    diar_audio.mkdir(parents=True, exist_ok=True)
    ts_audio.mkdir(parents=True, exist_ok=True)
    _download_annotations(ann_dir)

    diar_entries: list[dict] = []
    ts_entries: list[dict] = []

    for m in meeting_ids:
        dest = diar_audio / f"{m}.Mix-Headset.wav"
        if not dest.exists() and download_audio:
            try:
                _util.download(AUDIO_URL_TEMPLATE.format(m=m), dest)
            except Exception as e:  # noqa: BLE001
                print(f"  [SKIP] {m}: audio download failed: {e}")
                continue
        if not dest.exists():
            print(f"  [SKIP] {m}: no audio")
            continue

        duration = round(_util.ffprobe_duration(dest), 2)
        segs = _parse_segments(m, ann_dir)
        if segs:
            diar_entries.append({
                "id": f"ami-{m}",
                "audio_path": _util.rel_to_data(dest),
                "speakers": [{"speaker": s, "start": round(a, 3), "end": round(b, 3)}
                             for a, b, s in segs],
                "speaker_transcripts": _speaker_transcripts(m, ann_dir),
                "duration_s": duration,
            })

        words = _parse_words(m, ann_dir)
        if words:
            # Reuse the diarization audio for timestamps: symlink to avoid a
            # second multi-GB copy, falling back to a hard copy if symlinks fail.
            ts_dest = ts_audio / f"{m}.Mix-Headset.wav"
            if not ts_dest.exists():
                try:
                    ts_dest.symlink_to(dest.resolve())
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(dest, ts_dest)
            ts_entries.append({
                "id": f"ami-{m}",
                "audio_path": _util.rel_to_data(ts_dest),
                "words": words,
                "duration_s": duration,
            })
        print(f"  + {m}  segments={len(segs)}  words={len(words)}  dur={duration:.0f}s")

    _util.write_jsonl(
        config.manifest_path("diarization", "diarization_ami"), diar_entries,
        header=f"Diarization AMI (Mix-Headset) | collar={config.DER_COLLAR_S}s "
               f"| frozen {config.SUITE_VERSION} | n={len(diar_entries)}")
    _util.write_jsonl(
        config.manifest_path("timestamps", "timestamps_ami"), ts_entries,
        header=f"Timestamp ground truth AMI words | frozen {config.SUITE_VERSION} "
               f"| n={len(ts_entries)}")


def prepare_sdm(meeting_ids: list[str]) -> None:
    """AMI-SDM (Array1-01 + official RTTM) -> manifests/diarization/diarization_ami_sdm.jsonl."""
    audio_dir = config.audio_dir("diarization")
    rttm_dir = config.DATA_ROOT / "diarization" / "rttm"
    entries: list[dict] = []
    for m in meeting_ids:
        dest = audio_dir / f"{m}.Array1-01.wav"
        rttm = rttm_dir / f"{m}.rttm"
        uem = rttm_dir / f"{m}.uem"
        try:
            _util.download(SDM_AUDIO_URL_TEMPLATE.format(m=m), dest)
            _util.download(SDM_RTTM_URL_TEMPLATE.format(m=m), rttm)
            _util.download(SDM_UEM_URL_TEMPLATE.format(m=m), uem)
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] {m}: download failed: {e}")
            continue
        segs = []
        for line in rttm.read_text().splitlines():
            f = line.split()
            if len(f) >= 8 and f[0] == "SPEAKER":
                start, dur = float(f[3]), float(f[4])
                segs.append({"speaker": f[7], "start": round(start, 3),
                             "end": round(start + dur, 3)})
        segs.sort(key=lambda s: s["start"])
        entries.append({
            "id": f"ami-{m}",
            "audio_path": _util.rel_to_data(dest),
            "speakers": segs,
            "duration_s": round(float(uem.read_text().split()[3]), 3),
        })
        print(f"  + {m} (SDM)  segments={len(segs)}")
    _util.write_jsonl(
        config.manifest_path("diarization", "diarization_ami_sdm"), entries,
        header=f"AMI-SDM (single distant mic, Array1-01) + official RTTM refs "
               f"| overlap-aware DER | n={len(entries)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare AMI diarization + timestamp data")
    ap.add_argument("--meetings", type=int, default=20)
    ap.add_argument("--no-download-audio", action="store_true")
    ap.add_argument("--sdm-meetings", type=int, default=len(AMI_SDM_TEST_SET),
                    help="how many AMI-SDM meetings to prepare (0 to skip)")
    args = ap.parse_args()
    n = min(args.meetings, len(AMI_SC_TEST_SET))
    prepare(AMI_SC_TEST_SET[:n], download_audio=not args.no_download_audio)
    if args.sdm_meetings:
        prepare_sdm(AMI_SDM_TEST_SET[: args.sdm_meetings])


if __name__ == "__main__":
    main()
