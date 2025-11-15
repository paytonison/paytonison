use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs;
use std::path::Path;

const DICTIONARY_ENTRIES: [(&str, &str); 9] = [
    (
        "antidisestablishmentarianism",
        "Oxford English Dictionary (paraphrased): opposition to the withdrawal of state support or recognition from an established church.",
    ),
    (
        "hypermetamorphosis",
        "Oxford English Dictionary (paraphrased): a kind of insect development marked by distinctly different successive larval stages.",
    ),
    (
        "biblioklept",
        "Oxford English Dictionary (paraphrased): a person who steals books; a book thief.",
    ),
    (
        "defenestration",
        "Oxford English Dictionary (paraphrased): the act of throwing someone or something out of a window.",
    ),
    (
        "absquatulate",
        "Oxford English Dictionary (paraphrased): to depart abruptly; to abscond with comic haste.",
    ),
    (
        "cattywampus",
        "Oxford English Dictionary (paraphrased): askew or awry; positioned diagonally in a delightfully unruly fashion.",
    ),
    (
        "transmogrification",
        "Oxford English Dictionary (paraphrased): a transformation, especially one that is startling or magical.",
    ),
    (
        "sesquipedalian",
        "Oxford English Dictionary (paraphrased): characterized by or fond of using long words.",
    ),
    (
        "kerfuffle",
        "Oxford English Dictionary (paraphrased): a commotion or fuss, especially one caused by conflicting opinions.",
    ),
];

const PREFIXES: [&str; 11] = [
    "hyper", "trans", "inter", "sesqui", "anti", "ab", "de", "dis", "ultra", "pseudo", "super",
];

const SUFFIXES: [&str; 17] = [
    "arianism",
    "ification",
    "mogrification",
    "ulation",
    "arian",
    "ation",
    "mentation",
    "iasis",
    "iosis",
    "osis",
    "ulate",
    "icism",
    "ism",
    "ious",
    "ian",
    "ness",
    "ment",
];

const ROOT_PATTERNS: [&str; 21] = [
    "antidisestablish",
    "establish",
    "transmogr",
    "cattywampus",
    "sesquiped",
    "biblio",
    "bibli",
    "hyper",
    "mogr",
    "meta",
    "morph",
    "klept",
    "fenestr",
    "squat",
    "wampus",
    "pedal",
    "ped",
    "kerfuffle",
    "fic",
    "ruffle",
    "fuffle",
];

const SENSORY_KEYWORDS: [&str; 12] = [
    "light", "bright", "glow", "sound", "tone", "taste", "touch", "smell", "colour", "color",
    "hear", "see",
];
const ABSTRACT_KEYWORDS: [&str; 12] = [
    "state",
    "quality",
    "ability",
    "capacity",
    "process",
    "condition",
    "power",
    "toughness",
    "recovery",
    "change",
    "time",
    "action",
];

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dictionary = build_dictionary();

    let words = if let Some(path) = args.get(1) {
        let expanded = expand_tilde(path);
        read_words_from_file(Path::new(&expanded))?
    } else {
        dictionary.keys().map(|word| (*word).to_string()).collect()
    };

    if words.is_empty() {
        println!(
            "No words supplied; provide a newline separated list or rely on the built-in sample."
        );
        return Ok(());
    }

    println!("Processing {} words...", words.len());

    let mut embeddings: HashMap<MorphemeKey, EmbeddingAccumulator> = HashMap::new();
    let mut used_dictionary_entries = 0;
    let mut dimensions = None;

    for word in words {
        let canonical = word.trim().to_lowercase();
        if canonical.is_empty() {
            continue;
        }

        match dictionary.get(canonical.as_str()) {
            Some(definition) => {
                used_dictionary_entries += 1;
                let morphemes = segment_into_morphemes(&canonical);
                if morphemes.is_empty() {
                    println!("- {canonical}: no morphemic chunks produced by the segmenter");
                    continue;
                }

                let features = definition_to_features(definition);
                dimensions.get_or_insert(features.len());

                let breakdown = morphemes
                    .iter()
                    .map(|morpheme| morpheme.display())
                    .collect::<Vec<_>>()
                    .join(" + ");

                println!("- {canonical}: {breakdown}");
                println!("  definition: {definition}");

                for morpheme in &morphemes {
                    let entry = embeddings
                        .entry(MorphemeKey::from(morpheme))
                        .or_insert_with(|| EmbeddingAccumulator::new(features.len()));
                    entry.add(&features);
                }
            }
            None => {
                println!("- {canonical}: no Oxford entry available in the bundled data");
            }
        }
    }

    if used_dictionary_entries == 0 {
        println!(
            "No dictionary entries matched the provided words. Try using the bundled examples."
        );
        return Ok(());
    }

    println!();
    let dims = dimensions.unwrap_or(0);
    println!(
        "Derived morpheme embedding matrix ({} morphemes × {} features):",
        embeddings.len(),
        dims
    );

    let mut sorted: Vec<(MorphemeKey, Vec<f32>)> = embeddings
        .into_iter()
        .map(|(phoneme, acc)| (phoneme, acc.mean()))
        .collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    for (key, vector) in sorted {
        let pretty = vector
            .iter()
            .map(|value| format!("{value:.3}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  {:<22} -> [{pretty}]", key.describe());
    }

    Ok(())
}

fn build_dictionary() -> BTreeMap<&'static str, &'static str> {
    DICTIONARY_ENTRIES.iter().cloned().collect()
}

fn read_words_from_file(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    let words = contents
        .lines()
        .map(|line| line.split('#').next().unwrap_or(""))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect();
    Ok(words)
}

fn segment_into_morphemes(word: &str) -> Vec<Morpheme> {
    let cleaned: String = word
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if cleaned.is_empty() {
        return Vec::new();
    }

    let mut prefix_matches: Vec<&str> = Vec::new();
    let mut working = cleaned.as_str();

    loop {
        if let Some(prefix) = PREFIXES
            .iter()
            .find(|candidate| working.starts_with(*candidate))
        {
            if prefix.len() >= working.len() {
                break;
            }
            prefix_matches.push(*prefix);
            working = &working[prefix.len()..];
        } else {
            break;
        }
    }

    let mut suffix_matches: Vec<&str> = Vec::new();
    let mut core = working;
    loop {
        if let Some(suffix) = SUFFIXES.iter().find(|candidate| core.ends_with(*candidate)) {
            if suffix.len() >= core.len() {
                break;
            }
            suffix_matches.push(*suffix);
            core = &core[..core.len() - suffix.len()];
        } else {
            break;
        }
    }

    let mut segments: Vec<Morpheme> = prefix_matches
        .into_iter()
        .map(|prefix| Morpheme::new(MorphemeKind::Prefix, prefix))
        .collect();

    let mut root_segments = decompose_root_segments(core);
    if root_segments.is_empty() {
        if !core.is_empty() {
            root_segments.push(Morpheme::new(MorphemeKind::Root, core));
        } else {
            root_segments.push(Morpheme::new(MorphemeKind::Root, cleaned.as_str()))
        }
    }
    segments.append(&mut root_segments);

    suffix_matches.reverse();
    segments.extend(
        suffix_matches
            .into_iter()
            .map(|suffix| Morpheme::new(MorphemeKind::Suffix, suffix)),
    );

    segments
}

fn decompose_root_segments(core: &str) -> Vec<Morpheme> {
    if core.is_empty() {
        return Vec::new();
    }

    let mut remainder = core;
    let mut roots = Vec::new();

    while !remainder.is_empty() {
        if let Some(pattern) = ROOT_PATTERNS
            .iter()
            .find(|candidate| remainder.starts_with(*candidate))
        {
            let slice = *pattern;
            let len = slice.len();
            if len == 0 {
                break;
            }
            roots.push(Morpheme::new(MorphemeKind::Root, slice));
            remainder = &remainder[len..];
            continue;
        }

        if remainder.len() <= 4 {
            roots.push(Morpheme::new(MorphemeKind::Root, remainder));
            break;
        }

        let step = 4usize.min(remainder.len());
        let (chunk, rest) = remainder.split_at(step);
        roots.push(Morpheme::new(MorphemeKind::Root, chunk));
        remainder = rest;
    }

    roots
}

fn definition_to_features(definition: &str) -> Vec<f32> {
    let tokens: Vec<String> = definition
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect();

    if tokens.is_empty() {
        return vec![0.0; 5];
    }

    let word_count = tokens.len() as f32;
    let total_chars: usize = tokens.iter().map(|token| token.len()).sum();
    let avg_word_length = total_chars as f32 / word_count;

    let sensory_hits = tokens
        .iter()
        .filter(|token| SENSORY_KEYWORDS.contains(&token.as_str()))
        .count() as f32;

    let abstract_hits = tokens
        .iter()
        .filter(|token| ABSTRACT_KEYWORDS.contains(&token.as_str()))
        .count() as f32;

    let unique_tokens = tokens
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len() as f32;

    let multi_syllable_estimate = tokens.iter().filter(|token| token.len() >= 7).count() as f32;

    vec![
        word_count,
        avg_word_length,
        sensory_hits / word_count,
        abstract_hits / word_count,
        unique_tokens / word_count.max(1.0) + multi_syllable_estimate / word_count,
    ]
}

#[derive(Debug, Clone)]
struct Morpheme {
    kind: MorphemeKind,
    text: String,
}

impl Morpheme {
    fn new(kind: MorphemeKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
        }
    }

    fn display(&self) -> String {
        format!("{}({})", self.kind.label(), self.text)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum MorphemeKind {
    Prefix,
    Root,
    Suffix,
}

impl MorphemeKind {
    fn label(self) -> &'static str {
        match self {
            MorphemeKind::Prefix => "prefix",
            MorphemeKind::Root => "root",
            MorphemeKind::Suffix => "suffix",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct MorphemeKey {
    kind: MorphemeKind,
    text: String,
}

impl MorphemeKey {
    fn describe(&self) -> String {
        format!("{}:{}", self.kind.label(), self.text)
    }
}

impl From<&Morpheme> for MorphemeKey {
    fn from(morpheme: &Morpheme) -> Self {
        Self {
            kind: morpheme.kind,
            text: morpheme.text.clone(),
        }
    }
}

#[derive(Debug)]
struct EmbeddingAccumulator {
    sum: Vec<f32>,
    count: usize,
}

impl EmbeddingAccumulator {
    fn new(dimensions: usize) -> Self {
        Self {
            sum: vec![0.0; dimensions],
            count: 0,
        }
    }

    fn add(&mut self, vector: &[f32]) {
        if self.sum.len() != vector.len() {
            self.sum.resize(vector.len(), 0.0);
        }

        for (slot, value) in self.sum.iter_mut().zip(vector.iter()) {
            *slot += value;
        }
        self.count += 1;
    }

    fn mean(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.sum.len()];
        }
        let scale = self.count as f32;
        self.sum.iter().map(|value| value / scale).collect()
    }
}

fn expand_tilde(input: &str) -> String {
    if let Some(stripped) = input.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}/{}", home.trim_end_matches('/'), stripped);
        }
    }
    input.to_string()
}
