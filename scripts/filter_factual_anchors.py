"""Filter factual anchors to keep only referentially stable named entities.

Keeps: PERSON, ORG, GPE, LOC, WORK_OF_ART, EVENT, NORP, FACILITY, PRODUCT, LAW, LANGUAGE
Drops: DATE, CARDINAL, MONEY, PERCENT, ORDINAL, QUANTITY, TIME, generic descriptors, numbers

Uses GPT-4o-mini to classify ambiguous spans in batches.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verl.augment.openai_client import OpenAIClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRIVIALLY_DROP = re.compile(
    r"^("
    r"[#$€£¥₹]"             # currency/hex color prefixes
    r"|[-+]?\d[\d,.]*%?"     # plain numbers, negatives, percentages
    r"|v?\d+\.\d+[\w.-]*"   # version strings
    r"|\d+\s*(st|nd|rd|th)?" # bare ordinals
    r"|about\s+\d"           # "about 3"
    r"|approximately\s+\d"
    r"|around\s+\d"
    r"|over\s+\d"
    r"|under\s+\d"
    r"|more than\s+\d"
    r"|less than\s+\d"
    r"|at least\s+\d"
    r")$",
    re.IGNORECASE,
)

TRIVIALLY_DROP_PATTERNS = [
    re.compile(r"^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$", re.IGNORECASE),
    re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}$", re.IGNORECASE),
    re.compile(r"^\d{4}s?$"),                        # years like 1794, 1990s
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),        # dates like 3/15/2020
    re.compile(r"^[$€£¥₹][\d,.]+(\s*(million|billion|trillion|thousand|hundred|crores?|lakhs?))?$", re.IGNORECASE),
    re.compile(r"^[\d,.]+\s*(million|billion|trillion|thousand|hundred|crores?|lakhs?|percent|%|feet|meters|km|miles|kg|pounds|tons|acres|hectares|square\s+\w+|cubic\s+\w+)$", re.IGNORECASE),
    re.compile(r"^\d+[\s-]+(year|month|day|week|hour|minute|second)s?$", re.IGNORECASE),
    re.compile(r"^(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)[-\s]*(year|month|day|week|hour|minute|second)?s?$", re.IGNORECASE),
    re.compile(r"^(a|an|the|his|her|its|their|they|them|he|she|it|we|us|our|this|that|these|those|some|many|few|most|all|any|each|every|both|either|neither|no|none|other|another|such|same|different|various|several|much|more|less|least|enough|certain|own|whole|entire)$", re.IGNORECASE),
]

GENERIC_DESCRIPTORS = re.compile(
    r"^(woman|man|women|men|people|person|child|children|boy|girl|"
    r"first|second|third|fourth|fifth|last|next|previous|former|latter|"
    r"third.place|first.place|second.place|"
    r"year.end\s+\d|"
    r"web.based\s+\w+|"
    r"triplets|twins|"
    r"vice.chancellor)$",
    re.IGNORECASE,
)


def should_trivially_drop(anchor: str) -> bool:
    if TRIVIALLY_DROP.match(anchor):
        return True
    for pat in TRIVIALLY_DROP_PATTERNS:
        if pat.match(anchor):
            return True
    if GENERIC_DESCRIPTORS.match(anchor):
        return True
    if len(anchor) <= 1 and not anchor.isalpha():
        return True
    return False


BATCH_SIZE = 80

SYSTEM_PROMPT = """\
You are a named-entity classifier. For each span I give you, output KEEP or DROP.

KEEP spans that are referentially stable — they pick out one specific thing in the world:
- PERSON names (Robespierre, Marie Curie)
- ORG names (NASA, UNESCO, Manchester United)
- GPE / LOC (Paris, Lake Titicaca, Mount Everest)
- WORK_OF_ART (song titles, book titles, film titles, paper titles)
- EVENT names (Battle of Gettysburg, 2024 Olympics)
- NORP (French, Buddhist, Republican — nationalities/religions/political groups)
- FACILITY (Golden Gate Bridge)
- PRODUCT (iPhone, Boeing 747)
- LAW (First Amendment, GDPR)
- LANGUAGE names (Swahili, Mandarin)

DROP spans that are NOT referentially stable:
- Dates, years, time expressions (1794, March 2020, two days)
- Numbers, cardinals, ordinals (three, 1,078,692, twenty-third)
- Money amounts ($2.2 million, ₹55 crores)
- Percentages (0.012%, 45 percent)
- Quantities with units (1,142 feet, 10 km)
- Generic descriptors (woman, third place, vice-chancellor)
- Pronouns or determiners (they, his, the)
- Hashtags that are slogans not proper names (#blacklivesmatter → KEEP as movement name, but generic hashtags → DROP)
- Version strings (v0.4.0, v1.3-rc8)
- Color hex codes (#E58E73)

Respond with a JSON array of objects: [{"span": "...", "verdict": "KEEP"}, ...]
No other text."""


def classify_batch(client: OpenAIClient, spans: list[str]) -> dict[str, str]:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(spans))
    user_prompt = f"Classify these spans:\n{numbered}"

    raw = client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=4096,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    results = {}
    try:
        parsed = json.loads(raw)
        for entry in parsed:
            results[entry["span"]] = entry["verdict"].upper()
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Failed to parse LLM response, marking batch as KEEP for safety")
        for s in spans:
            results[s] = "KEEP"

    return results


def main():
    parser = argparse.ArgumentParser(description="Filter factual anchors to referentially stable named entities")
    parser.add_argument("input", type=Path, help="Path to factual anchors txt file (one per line)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: <input>_filtered.txt)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Spans per API call (default: {BATCH_SIZE})")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel API calls (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be dropped without calling the API")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        args.output = args.input.with_name(args.input.stem + "_filtered.txt")

    anchors = [line.strip() for line in args.input.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(anchors)} anchors from {args.input}")

    trivial_drops = []
    ambiguous = []
    for a in anchors:
        if should_trivially_drop(a):
            trivial_drops.append(a)
        else:
            ambiguous.append(a)

    print(f"Trivially dropped: {len(trivial_drops)} (numbers, dates, pronouns, etc.)")
    print(f"Need LLM classification: {len(ambiguous)}")

    if args.dry_run:
        print("--- Trivially dropped samples ---")
        for d in trivial_drops[:30]:
            print(f"  DROP: {d}")
        print("--- Ambiguous samples (would send to LLM) ---")
        for a in ambiguous[:30]:
            print(f"  ???:  {a}")
        print(f"Would keep at most {len(ambiguous)} after LLM filtering")
        return

    client = OpenAIClient(model=args.model)

    batches = []
    for i in range(0, len(ambiguous), args.batch_size):
        batches.append(ambiguous[i : i + args.batch_size])
    total_batches = len(batches)
    print(f"Will process {total_batches} batches with {args.num_workers} workers")

    kept = []
    llm_dropped = []
    completed = 0

    def _process_batch(batch_idx_and_spans):
        batch_idx, spans = batch_idx_and_spans
        return batch_idx, spans, classify_batch(client, spans)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(_process_batch, (i, batch)): i
            for i, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(futures):
            batch_idx, spans, verdicts = future.result()
            for span in spans:
                verdict = verdicts.get(span, "KEEP")
                if verdict == "KEEP":
                    kept.append(span)
                else:
                    llm_dropped.append(span)
            completed += 1
            print(f"Batch {batch_idx + 1}/{total_batches} done ({completed}/{total_batches} completed)")

    print(f"LLM kept: {len(kept)}, LLM dropped: {len(llm_dropped)}")
    print(f"Total: {len(anchors)} -> {len(kept)} ({len(anchors) - len(kept)} removed)")

    kept.sort()
    args.output.write_text("\n".join(kept) + "\n")
    print(f"Written to {args.output}")

    dropped_path = args.output.with_name(args.output.stem + "_dropped.txt")
    all_dropped = sorted(trivial_drops + llm_dropped)
    dropped_path.write_text("\n".join(all_dropped) + "\n")
    print(f"Dropped anchors written to {dropped_path}")


if __name__ == "__main__":
    main()
