# Erebus

Erebus is a tiny Rust adventure into morpheme land. It takes an Oxford-flavoured mini-dictionary of gloriously long words, splits every entry into prefixes, roots, and suffixes, and then distils each morpheme into a numeric embedding based on the definition text. The goal is to prototype how definition-driven features can be shared across word parts.

## Features
- Segments supplied words into morphemes using a handcrafted list of prefixes, suffixes, and root patterns.
- Generates a five-dimensional feature vector from each definition (length, sensory/abstract leaning, lexical variety).
- Averages those features across all appearances of a morpheme to give it an embedding.
- Prints an easy-to-read breakdown of every word plus the resulting morpheme matrix.
- Accepts an optional newline-separated word list (with `#` comments) or falls back to the bundled dictionary keys.

## Quick Start
1. Install a recent Rust toolchain (the project targets edition 2024).
2. Clone the repo and run the binary:

```bash
cargo run
```

Without arguments the program iterates over the nine built-in dictionary entries. To analyse a custom list, pass a path to a file containing one word per line (leading `~/` is expanded automatically):

```bash
cargo run -- examples/demo_words.txt
```

Lines starting with `#` are ignored so you can annotate your lists.

## Sample Output
```text
Processing 9 words...
- antidisestablishmentarianism: prefix(anti) + prefix(dis) + root(establish) + suffix(ment) + suffix(arianism)
  definition: Oxford English Dictionary (paraphrased): opposition to the withdrawal of state support or recognition from an established church.
...
Derived morpheme embedding matrix (25 morphemes × 5 features):
  prefix:anti            -> [17.000, 6.412, 0.000, 0.059, 1.471]
  root:establish         -> [17.000, 6.412, 0.000, 0.059, 1.471]
  suffix:arianism        -> [17.000, 6.412, 0.000, 0.059, 1.471]
  ...
```

## How It Works
- `src/main.rs` builds a `BTreeMap` of dictionary entries (`DICTIONARY_ENTRIES`) and performs all processing.
- `segment_into_morphemes` filters non ASCII characters, peels off known prefixes/suffixes, then searches for root patterns before falling back to fixed-width chunks.
- `definition_to_features` tokenises the definition into lowercase words and computes: word count, average token length, sensory word ratio, abstract word ratio, and a combined uniqueness/multi-syllable score.
- `EmbeddingAccumulator` collects feature vectors per morpheme and reports their mean as the final embedding.

## Extending The Experiment
- Add more entries to `DICTIONARY_ENTRIES` or load them from disk.
- Grow the prefix/suffix/root tables for better segmentation coverage.
- Swap in a richer feature extractor (POS tags, embeddings, etc.) to improve the morpheme vectors.
- Export the embedding matrix in a machine-friendly format (CSV, JSON) for downstream modelling.

Happy spelunking!

---

> Distributed under the MIT License.
