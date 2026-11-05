//! Wildcard and Levenshtein matching for NodeCraft tool search.
//!
//! Both utilities are Rust ports of CHOC text helpers
//! (<https://github.com/Tracktion/choc>, ISC licence; see
//! `THIRD_PARTY_NOTICES.md`):
//!
//! - [`wildcard_matches`] follows `choc::text::WildcardPattern`: `*` matches
//!   any run, `?` matches one character, and semicolons separate alternatives.
//! - [`levenshtein`] follows `choc::text::getLevenshteinDistance` and is used
//!   as the typo-tolerant fallback in tool search.

/// Match `text` against one or more wildcard patterns separated by `;`.
#[must_use]
pub fn wildcard_matches(pattern: &str, text: &str) -> bool {
    let text: Vec<char> = text.chars().collect();
    pattern
        .split(';')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .any(|part| {
            let pattern: Vec<char> = part.chars().collect();
            wildcard_matches_all(&text, &pattern)
        })
}

fn wildcard_matches_anywhere(source: &[char], pattern: &[char]) -> bool {
    let mut index = 0;
    while index < source.len() {
        if wildcard_matches_all(&source[index..], pattern) {
            return true;
        }
        index += 1;
    }
    false
}

fn wildcard_matches_all(source: &[char], pattern: &[char]) -> bool {
    let Some(&pattern_char) = pattern.first() else {
        return source.is_empty();
    };
    let pattern_rest = &pattern[1..];
    if pattern_char == '*' {
        return pattern_rest.is_empty() || wildcard_matches_anywhere(source, pattern_rest);
    }
    let Some(&source_char) = source.first() else {
        return false;
    };
    if pattern_char == '?' {
        return wildcard_matches_all(&source[1..], pattern_rest);
    }
    if !source_char.eq_ignore_ascii_case(&pattern_char) {
        return false;
    }
    wildcard_matches_all(&source[1..], pattern_rest)
}

/// Levenshtein edit distance: the minimum number of insertions, deletions and
/// substitutions needed to turn `a` into `b`.
///
/// This is a Rust port of CHOC's `choc::text::getLevenshteinDistance` and runs
/// in O(a.len() * b.len()) time with O(b.len()) working memory.
#[must_use]
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut current = vec![i + 1; b.len() + 1];
        for (j, &cb) in b.iter().enumerate() {
            current[j + 1] = if ca == cb {
                previous[j]
            } else {
                1 + previous[j].min(previous[j + 1]).min(current[j])
            };
        }
        previous = current;
    }
    previous[b.len()]
}

/// The closest Levenshtein distance from `needle` to a whole text or one of
/// its whitespace-separated words, case-insensitively.
#[must_use]
pub fn best_levenshtein(needle: &str, text: &str) -> usize {
    let needle = needle.to_lowercase();
    let text = text.to_lowercase();
    text.split_whitespace()
        .map(|word| levenshtein(&needle, word))
        .chain(std::iter::once(levenshtein(&needle, &text)))
        .min()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_matches_anywhere_and_honours_question_mark() {
        assert!(wildcard_matches("*box*", "Toolbox"));
        assert!(wildcard_matches("b?x", "box"));
        assert!(!wildcard_matches("b?x", "boxer"));
        assert!(wildcard_matches("box;torus", "Torus"));
    }

    #[test]
    fn levenshtein_handles_typos_and_words() {
        assert_eq!(levenshtein("boz", "box"), 1);
        assert_eq!(best_levenshtein("cylindr", "Cylinder"), 1);
        assert!(best_levenshtein("offset", "Offset Faces") <= 1);
    }
}
