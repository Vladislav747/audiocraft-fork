import argparse
import json
from pathlib import Path

import torch
import torchaudio
from audiocraft.models import MusicGen

PROMPTS = [
    {
        "description": "An epic and triumphant orchestral soundtrack featuring powerful brass and a sweeping string ensemble, driven by a fast march-like rhythm and an epic background choir, recorded with massive stadium reverb.",
        "general_mood": "Epic, heroic, triumphant, building tension",
        "genre_tags": ["Cinematic", "Orchestral", "Soundtrack"],
        "lead_instrument": "Powerful brass section (horns, trombones)",
        "accompaniment": "Sweeping string ensemble, heavy cinematic percussion, timpani",
        "tempo_and_rhythm": "Fast, driving, march-like rhythm",
        "vocal_presence": "Epic choir in the background (wordless chanting)",
        "production_quality": "High fidelity, wide stereo image, massive stadium reverb",
    },
    {
        "description": "A relaxing lo-fi hip-hop instrumental with a muffled electric piano playing jazz chords over a dusty vinyl crackle, deep sub-bass, and a slow boom-bap drum loop.",
        "general_mood": "Relaxing, nostalgic, chill, melancholic",
        "genre_tags": ["Lo-Fi Hip Hop", "Chillhop", "Instrumental"],
        "lead_instrument": "Muffled electric piano (Rhodes) playing jazz chords",
        "accompaniment": "Dusty vinyl crackle, deep sub-bass, soft boom-bap drum loop",
        "tempo_and_rhythm": "Slow, laid-back, swinging groove",
        "vocal_presence": "None",
        "production_quality": "Lo-Fi, vintage, warm tape saturation, slightly muffled high frequencies",
    },
    {
        "description": "An energetic progressive house dance track with a bright detuned synthesizer lead, pumping sidechain bass, and chopped vocal samples over a fast four-on-the-floor beat.",
        "general_mood": "Energetic, uplifting, party vibe, euphoric",
        "genre_tags": ["EDM", "Progressive House", "Dance"],
        "lead_instrument": "Bright, detuned synthesizer lead",
        "accompaniment": "Pumping sidechain bass, risers, crash cymbals",
        "tempo_and_rhythm": "Fast, driving, strict four-on-the-floor beat",
        "vocal_presence": "Chopped vocal samples used as a rhythmic instrument",
        "production_quality": "Modern, extremely loud, punchy, club-ready mix",
    },
    {
        "description": "An intimate acoustic folk instrumental featuring a fingerpicked acoustic guitar, light tambourine, and subtle upright bass, played in a gentle waltz-like rhythm.",
        "general_mood": "Intimate, warm, acoustic, peaceful",
        "genre_tags": ["Folk", "Acoustic", "Indie"],
        "lead_instrument": "Fingerpicked acoustic guitar",
        "accompaniment": "Light tambourine, subtle upright bass, distant ambient room sound",
        "tempo_and_rhythm": "Mid-tempo, gentle, waltz-like triple meter",
        "vocal_presence": "None",
        "production_quality": "Raw, organic, close-mic recording, natural room acoustics",
    },
    {
        "description": "A dark cyberpunk synthwave instrumental driven by an aggressive distorted analog bass synthesizer, arpeggiated synth plucks, and a retro 80s drum machine.",
        "general_mood": "Dark, futuristic, gritty, mysterious",
        "genre_tags": ["Synthwave", "Cyberpunk", "Darkwave"],
        "lead_instrument": "Aggressive, distorted analog bass synthesizer",
        "accompaniment": "Arpeggiated synth plucks, retro 80s drum machine (gated snare)",
        "tempo_and_rhythm": "Driving, mid-tempo, robotic precision",
        "vocal_presence": "None",
        "production_quality": "Retro-futuristic, heavy compression, synthetic, 80s aesthetic",
    },
]


def structured_prompt_to_text(prompt: dict) -> str:
    return (
        f"description: {prompt['description']}. "
        f"general_mood: {prompt['general_mood']}. "
        f"genre_tags: {', '.join(prompt['genre_tags'])}. "
        f"lead_instrument: {prompt['lead_instrument']}. "
        f"accompaniment: {prompt['accompaniment']}. "
        f"tempo_and_rhythm: {prompt['tempo_and_rhythm']}. "
        f"vocal_presence: {prompt['vocal_presence']}. "
        f"production_quality: {prompt['production_quality']}."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate 5 tracks from structured prompts.")
    parser.add_argument("--model", default="facebook/musicgen-small")
    parser.add_argument("--duration", type=int, default=12)
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--save-prompts-json",
        action="store_true",
        help="Save structured prompt JSON files near wav outputs",
    )
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MusicGen.get_pretrained(args.model, device=device)
    model.set_generation_params(duration=args.duration)

    prompt_texts = [structured_prompt_to_text(p) for p in PROMPTS]
    with torch.no_grad():
        wav = model.generate(prompt_texts).cpu()

    sample_rate = int(getattr(model, "sample_rate", 32000))
    for idx, item in enumerate(PROMPTS, start=1):
        wav_path = out_dir / f"prompt_{idx}.wav"
        torchaudio.save(str(wav_path), wav[idx - 1], sample_rate=sample_rate)
        if args.save_prompts_json:
            json_path = out_dir / f"prompt_{idx}.json"
            json_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {wav_path}")


if __name__ == "__main__":
    main()
