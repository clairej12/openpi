#!/usr/bin/env python3
"""
find_open_tasks.py

Scan the DROID dataset for episodes whose language instruction involves "open"-type tasks.
Counts and prints matching instructions, and saves a GIF of the first such episode.
"""

import re
import string
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

# ---------- Helpers ----------

def as_gif(images, path="open_example_episode.gif", fps=15):
    """Render the images as a GIF (default 15 Hz)."""
    if not images:
        return None
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved GIF to: {path}")
    return path


def normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


# Common "open" inflections & synonyms
OPEN_PATTERNS = [
    r"\bopen(?:ing|ed)?\b",
    r"\bunlatch(?:ing|ed)?\b",
    r"\bunseal(?:ing|ed)?\b",
    r"\buncap(?:ping|ped)?\b",
    r"\bunlock(?:ing|ed)?\b",
    r"\bunscrew(?:ing|ed)?\b",
    r"\bcrack(?:ing)? open\b",
    r"\blift (?:the )?lid\b",
    r"\bremove (?:the )?lid\b",
    r"\bpop open\b",
    r"\bpull open\b",
    r"\bslide open\b",
    r"\braise (?:the )?lid\b",
]
open_re = re.compile("|".join(OPEN_PATTERNS))


def extract_instruction_from_first_step(episode_steps):
    """Return the first non-empty instruction among the three variants from the first step."""
    for step in episode_steps:
        for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
            if key in step:
                v = step[key].numpy()
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8", errors="ignore")
                v = (v or "").strip()
                if v:
                    return v
        break
    return None


def is_open_task(instruction: str) -> bool:
    s = normalize(instruction or "")
    return bool(open_re.search(s))


# ---------- Main ----------

def main():
    print("Loading DROID dataset (this may take a while)...")
    ds = tfds.load("droid_100", data_dir="/media/volume/models_and_data/DROID/", split="train")

    open_count = 0
    open_instructions = []
    gif_made = False

    for ep_idx, episode in enumerate(ds):
        instr = extract_instruction_from_first_step(episode["steps"])
        if instr and is_open_task(instr):
            open_count += 1
            open_instructions.append(instr)

            # Make a GIF for the first matching episode
            if not gif_made:
                print(f"\nCreating GIF for episode #{ep_idx}:\n  Instruction: {instr}")
                images = []
                for step in episode["steps"]:
                    img = np.concatenate(
                        (
                            step["observation"]["exterior_image_1_left"].numpy(),
                            step["observation"]["exterior_image_2_left"].numpy(),
                            step["observation"]["wrist_image_left"].numpy(),
                        ),
                        axis=1,
                    )
                    images.append(Image.fromarray(img))
                as_gif(images, "droid_sanity/open_episode.gif")
                gif_made = True

    print(f"\nFound {open_count} 'open'-type episodes.")
    uniq = sorted(set(open_instructions))
    print("\nSample unique instructions (up to 50):")
    for s in uniq[:50]:
        print("-", s)


if __name__ == "__main__":
    main()