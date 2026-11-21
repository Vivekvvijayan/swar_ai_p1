import os
import re
import wave
import uuid
import random
import numpy as np

# Carnatic Swara Frequency Multipliers relative to Tonic S (Shadjam = 1.0)
SWARA_RATIOS = {
    "S": 1.0,
    "R1": 16.0 / 15.0,    # Suddha Rishabham (~234.7 Hz)
    "R2": 9.0 / 8.0,      # Chatusruti Rishabham (247.5 Hz)
    "R3": 6.0 / 5.0,      # Satsruti Rishabham (264.0 Hz)
    "G1": 9.0 / 8.0,      # Suddha Gandharam
    "G2": 6.0 / 5.0,      # Sadharana Gandharam (264.0 Hz)
    "G3": 5.0 / 4.0,      # Antara Gandharam (275.0 Hz)
    "M1": 4.0 / 3.0,      # Suddha Madhyamam (~293.3 Hz)
    "M2": 45.0 / 32.0,    # Prati Madhyamam (~309.4 Hz)
    "P": 3.0 / 2.0,       # Panchamam (330.0 Hz)
    "D1": 8.0 / 5.0,      # Suddha Dhaivatam (352.0 Hz)
    "D2": 5.0 / 3.0,      # Chatusruti Dhaivatam (~366.7 Hz)
    "D3": 9.0 / 5.0,      # Satsruti Dhaivatam (396.0 Hz)
    "N1": 5.0 / 3.0,      # Suddha Nishadham
    "N2": 9.0 / 5.0,      # Kaisiki Nishadham (396.0 Hz)
    "N3": 15.0 / 8.0,     # Kakali Nishadham (412.5 Hz)
    "S_HIGH": 2.0,        # Tara Shadjam (440.0 Hz)
    "R1_HIGH": 2.0 * (16.0 / 15.0),
    "R2_HIGH": 2.0 * (9.0 / 8.0),
    "G2_HIGH": 2.0 * (6.0 / 5.0),
    "G3_HIGH": 2.0 * (5.0 / 4.0),
    "M1_HIGH": 2.0 * (4.0 / 3.0),
    "P_LOW": 0.5 * 1.5,   # Mandra Panchamam (165.0 Hz)
    "D1_LOW": 0.5 * (8.0 / 5.0),
    "D2_LOW": 0.5 * (5.0 / 3.0),
    "N2_LOW": 0.5 * (9.0 / 5.0),
    "N3_LOW": 0.5 * (15.0 / 8.0),
}

# Unicode display mappings for swaras
SWARA_DISPLAY = {
    "S": "S", "R1": "R₁", "R2": "R₂", "R3": "R₃",
    "G1": "G₁", "G2": "G₂", "G3": "G₃",
    "M1": "M₁", "M2": "M₂",
    "P": "P", "D1": "D₁", "D2": "D₂", "D3": "D₃",
    "N1": "N₁", "N2": "N₂", "N3": "N₃",
    "S_HIGH": "Ṡ", "R1_HIGH": "Ṙ₁", "R2_HIGH": "Ṙ₂",
    "G2_HIGH": "Ġ₂", "G3_HIGH": "Ġ₃", "M1_HIGH": "Ṁ₁",
    "P_LOW": "P̣", "D1_LOW": "Ḍ₁", "D2_LOW": "Ḍ₂",
    "N2_LOW": "Ṇ₂", "N3_LOW": "Ṇ₃",
    "-": "-"
}

# Raga Scales Knowledge Base (Arohanam / Avarohanam)
RAGA_SCALES = {
    "mohanam": {
        "arohanam": ["S", "R2", "G3", "P", "D2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "D2", "P", "G3", "R2", "S"],
        "vadi": "G3", "samvadi": "D2"
    },
    "kalyani": {
        "arohanam": ["S", "R2", "G3", "M2", "P", "D2", "N3", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N3", "D2", "P", "M2", "G3", "R2", "S"],
        "vadi": "G3", "samvadi": "N3"
    },
    "shankarabharanam": {
        "arohanam": ["S", "R2", "G3", "M1", "P", "D2", "N3", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N3", "D2", "P", "M1", "G3", "R2", "S"],
        "vadi": "G3", "samvadi": "P"
    },
    "sankarabharanam": {
        "arohanam": ["S", "R2", "G3", "M1", "P", "D2", "N3", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N3", "D2", "P", "M1", "G3", "R2", "S"],
        "vadi": "G3", "samvadi": "P"
    },
    "mayamalavagowla": {
        "arohanam": ["S", "R1", "G3", "M1", "P", "D1", "N3", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N3", "D1", "P", "M1", "G3", "R1", "S"],
        "vadi": "G3", "samvadi": "D1"
    },
    "hamsadhwani": {
        "arohanam": ["S", "R2", "G3", "P", "N3", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N3", "P", "G3", "R2", "S"],
        "vadi": "G3", "samvadi": "P"
    },
    "hindolam": {
        "arohanam": ["S", "G2", "M1", "D1", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D1", "M1", "G2", "S"],
        "vadi": "M1", "samvadi": "S"
    },
    "abheri": {
        "arohanam": ["S", "G2", "M1", "P", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D2", "P", "M1", "G2", "R2", "S"],
        "vadi": "G2", "samvadi": "P"
    },
    "kharaharapriya": {
        "arohanam": ["S", "R2", "G2", "M1", "P", "D2", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D2", "P", "M1", "G2", "R2", "S"],
        "vadi": "R2", "samvadi": "P"
    },
    "karaharapriya": {
        "arohanam": ["S", "R2", "G2", "M1", "P", "D2", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D2", "P", "M1", "G2", "R2", "S"],
        "vadi": "R2", "samvadi": "P"
    },
    "todi": {
        "arohanam": ["S", "R1", "G2", "M1", "P", "D1", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D1", "P", "M1", "G2", "R1", "S"],
        "vadi": "G2", "samvadi": "D1"
    },
    "thodi": {
        "arohanam": ["S", "R1", "G2", "M1", "P", "D1", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D1", "P", "M1", "G2", "R1", "S"],
        "vadi": "G2", "samvadi": "D1"
    },
    "bhairavi": {
        "arohanam": ["S", "R2", "G2", "M1", "P", "D2", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D1", "P", "M1", "G2", "R2", "S"],
        "vadi": "P", "samvadi": "R2"
    },
    "madhyamavati": {
        "arohanam": ["S", "R2", "M1", "P", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "P", "M1", "R2", "S"],
        "vadi": "R2", "samvadi": "P"
    },
    "charukesi": {
        "arohanam": ["S", "R2", "G3", "M1", "P", "D1", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D1", "P", "M1", "G3", "R2", "S"],
        "vadi": "M1", "samvadi": "S"
    },
    "revati": {
        "arohanam": ["S", "R1", "M1", "P", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "P", "M1", "R1", "S"],
        "vadi": "P", "samvadi": "S"
    },
    "chakravakam": {
        "arohanam": ["S", "R1", "G3", "M1", "P", "D2", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D2", "P", "M1", "G3", "R1", "S"],
        "vadi": "G3", "samvadi": "D2"
    },
    "kambhoji": {
        "arohanam": ["S", "R2", "G3", "M1", "P", "D2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D2", "P", "M1", "G3", "R2", "S"],
        "vadi": "D2", "samvadi": "G3"
    },
    "harikambhoji": {
        "arohanam": ["S", "R2", "G3", "M1", "P", "D2", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D2", "P", "M1", "G3", "R2", "S"],
        "vadi": "G3", "samvadi": "N2"
    },
    "natabhairavi": {
        "arohanam": ["S", "R2", "G2", "M1", "P", "D1", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D1", "P", "M1", "G2", "R2", "S"],
        "vadi": "G2", "samvadi": "D1"
    },
    "sahana": {
        "arohanam": ["S", "R2", "G3", "M1", "P", "M1", "D2", "N2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "S_HIGH", "D2", "N2", "D2", "P", "M1", "G3", "M1", "R2", "G3", "R2", "S"],
        "vadi": "R2", "samvadi": "P"
    },
    "bilahari": {
        "arohanam": ["S", "R2", "G3", "P", "D2", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N3", "D2", "P", "M1", "G3", "R2", "S"],
        "vadi": "G3", "samvadi": "D2"
    },
    "kapi": {
        "arohanam": ["S", "R2", "M1", "P", "N3", "S_HIGH"],
        "avarohanam": ["S_HIGH", "N2", "D2", "N2", "P", "M1", "G2", "R2", "S"],
        "vadi": "P", "samvadi": "R2"
    }
}


def get_raga_scale(raga_name: str):
    clean = re.sub(r"[^a-zA-Z]", "", raga_name).lower()
    if clean in RAGA_SCALES:
        return RAGA_SCALES[clean]

    for key, scale in RAGA_SCALES.items():
        if key in clean or clean in key:
            return scale

    return RAGA_SCALES["shankarabharanam"]


def synthesize_note(freq, duration_sec, sample_rate=44100):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # Harmonic additive synthesis (Carnatic flute/veena timbre)
    harmonic1 = np.sin(2 * np.pi * freq * t)
    harmonic2 = 0.45 * np.sin(2 * np.pi * (2 * freq) * t)
    harmonic3 = 0.20 * np.sin(2 * np.pi * (3 * freq) * t)
    harmonic4 = 0.08 * np.sin(2 * np.pi * (4 * freq) * t)
    
    # Subtle microtonal vibrato/gamaka oscillation (5 Hz modulation)
    vibrato = 0.02 * np.sin(2 * np.pi * 5.0 * t)
    wave_signal = harmonic1 + harmonic2 + harmonic3 + harmonic4 + vibrato

    # Natural ADSR Envelope
    total_samples = len(t)
    attack_samples = int(0.04 * sample_rate)
    release_samples = int(0.06 * sample_rate)
    
    envelope = np.ones(total_samples, dtype=np.float32)
    if attack_samples > 0 and attack_samples < total_samples:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples > 0 and release_samples < total_samples:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    return wave_signal * envelope


def generate_tanpura_drone(duration_sec, tonic_freq=220.0, sample_rate=44100):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Tanpura drone: S (110 Hz), P (165 Hz), S (220 Hz)
    s_low = 0.08 * np.sin(2 * np.pi * (tonic_freq * 0.5) * t)
    p_mid = 0.06 * np.sin(2 * np.pi * (tonic_freq * 0.75) * t)
    s_mid = 0.05 * np.sin(2 * np.pi * tonic_freq * t)
    return s_low + p_mid + s_mid


def render_phrase_to_wav(swara_sequence, note_durations, output_path, tonic_freq=220.0, sample_rate=44100):
    total_audio = []
    for swara, dur in zip(swara_sequence, note_durations):
        if swara == "-":
            silence = np.zeros(int(sample_rate * dur), dtype=np.float32)
            total_audio.append(silence)
        else:
            freq_mult = SWARA_RATIOS.get(swara, 1.0)
            note_audio = synthesize_note(tonic_freq * freq_mult, dur, sample_rate)
            total_audio.append(note_audio)

    combined_notes = np.concatenate(total_audio)
    total_duration = len(combined_notes) / sample_rate

    # Layer warm Tanpura drone
    tanpura = generate_tanpura_drone(total_duration, tonic_freq, sample_rate)
    final_audio = combined_notes + tanpura[:len(combined_notes)]

    # Normalize to prevent clipping (-1 dBFS)
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = (final_audio / max_val) * 0.88

    # Convert to 16-bit PCM
    audio_int16 = (final_audio * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)       # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return total_duration


def format_swara_display(swara_list):
    return " ".join(SWARA_DISPLAY.get(s, s) for s in swara_list)


def generate_probabilistic_phrase(scale, min_length=9, max_length=13, favor_upper=False):
    """
    Stochastic Markov phrase generator that creates unique Carnatic melodic sequences
    honoring Arohanam/Avarohanam grammar on every single invocation.
    """
    arohanam = scale["arohanam"]
    avarohanam = scale["avarohanam"]
    vadi = scale.get("vadi", arohanam[min(2, len(arohanam)-1)])
    samvadi = scale.get("samvadi", arohanam[min(4, len(arohanam)-1)])

    target_length = random.randint(min_length, max_length)
    phrase = []
    durations = []

    # Start note: S, Vadi, or Samvadi
    if favor_upper:
        curr = random.choice([samvadi, arohanam[-2]])
    else:
        curr = random.choice(["S", vadi])

    phrase.append(curr)
    durations.append(random.choice([0.45, 0.6]))

    for i in range(target_length - 2):
        r = random.random()

        if favor_upper and i == (target_length // 2):
            curr = "S_HIGH"
            dur = random.choice([0.8, 1.0])
        elif r < 0.40:
            # Ascend along Arohanam
            if curr in arohanam:
                idx = arohanam.index(curr)
                next_idx = min(len(arohanam) - 1, idx + 1)
                curr = arohanam[next_idx]
            else:
                curr = random.choice(arohanam)
            dur = random.choice([0.35, 0.45, 0.55])
        elif r < 0.75:
            # Descend along Avarohanam
            if curr in avarohanam:
                idx = avarohanam.index(curr)
                next_idx = min(len(avarohanam) - 1, idx + 1)
                curr = avarohanam[next_idx]
            else:
                curr = random.choice(avarohanam)
            dur = random.choice([0.40, 0.50, 0.65])
        elif r < 0.88:
            # Linger on Vadi / Samvadi resting swara (Karvai)
            curr = random.choice([vadi, samvadi])
            dur = random.choice([0.75, 0.95])
        else:
            # Janta swara (repeated note)
            dur = 0.35

        phrase.append(curr)
        durations.append(dur)

        # Musical breath / pause
        if random.random() < 0.18 and 2 < i < target_length - 4:
            phrase.append("-")
            durations.append(0.25)

    # Resolution cadence back to S
    phrase.append("S")
    durations.append(1.2)

    return phrase, durations


def generate_tunes_for_raga(raga_name: str, output_dir: str, base_url: str = ""):
    scale = get_raga_scale(raga_name)
    arohanam = scale["arohanam"]
    avarohanam = scale["avarohanam"]
    vadi = scale.get("vadi", arohanam[min(2, len(arohanam)-1)])

    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", raga_name).lower()
    run_id = uuid.uuid4().hex[:5]  # Ensures unique filenames and cache-busting per generation
    tunes = []

    # 1. Scale Cadence (Arohanam & Avarohanam with expressive dynamic durations)
    phrase1_swaras = arohanam + ["-"] + avarohanam
    phrase1_durs = [random.choice([0.4, 0.5]) for _ in arohanam] + [0.3] + [random.choice([0.4, 0.5]) for _ in avarohanam]
    phrase1_durs[len(arohanam)-1] = 0.9  # Hold upper Shadjam
    phrase1_durs[-1] = 1.25              # Resolving tonic note

    file1 = f"{clean_name}_tune_1_{run_id}.wav"
    path1 = os.path.join(output_dir, file1)
    dur1 = render_phrase_to_wav(phrase1_swaras, phrase1_durs, path1)
    tunes.append({
        "id": 1,
        "title": "Arohana - Avarohana Cadence",
        "swaras": f"{format_swara_display(arohanam)}  |  {format_swara_display(avarohanam)}",
        "description": f"Foundational ascending and descending scale cadence introducing the key swaras of {raga_name}.",
        "filename": file1,
        "audioUrl": f"{base_url}/api/audio/{file1}" if base_url else f"/api/audio/{file1}",
        "durationSeconds": round(dur1, 2)
    })

    # 2. Dynamic Madhya Sthayi Sanchara (Stochastically generated improvisation)
    phrase2_swaras, phrase2_durs = generate_probabilistic_phrase(scale, min_length=10, max_length=14, favor_upper=False)
    file2 = f"{clean_name}_tune_2_{run_id}.wav"
    path2 = os.path.join(output_dir, file2)
    dur2 = render_phrase_to_wav(phrase2_swaras, phrase2_durs, path2)
    tunes.append({
        "id": 2,
        "title": "Madhya Sthayi Sanchara",
        "swaras": format_swara_display(phrase2_swaras),
        "description": f"Dynamic melodic phrase in the middle octave highlighting the {vadi} swara with spontaneous variations.",
        "filename": file2,
        "audioUrl": f"{base_url}/api/audio/{file2}" if base_url else f"/api/audio/{file2}",
        "durationSeconds": round(dur2, 2)
    })

    # 3. Dynamic Tara Sthayi Sanchara (Upper Register Soaring Phrase)
    phrase3_swaras, phrase3_durs = generate_probabilistic_phrase(scale, min_length=9, max_length=12, favor_upper=True)
    file3 = f"{clean_name}_tune_3_{run_id}.wav"
    path3 = os.path.join(output_dir, file3)
    dur3 = render_phrase_to_wav(phrase3_swaras, phrase3_durs, path3)
    tunes.append({
        "id": 3,
        "title": "Tara Sthayi Sanchara",
        "swaras": format_swara_display(phrase3_swaras),
        "description": f"Soaring phrase touching Tara Shadjam (Ṡ) and gracefully cascading back to Adhara Shadjam (S).",
        "filename": file3,
        "audioUrl": f"{base_url}/api/audio/{file3}" if base_url else f"/api/audio/{file3}",
        "durationSeconds": round(dur3, 2)
    })

    return {
        "raga": raga_name,
        "count": len(tunes),
        "tunes": tunes
    }
