//! The labels file: which passcode each classifier output stands for.

use std::collections::HashMap;

use serde::Deserialize;
use thiserror::Error;

use crate::Passcode;

/// The passcode of each classifier output, by output index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Labels {
    passcodes: Vec<Passcode>,
}

impl Labels {
    /// Reads a labels file: a JSON object keyed by output index (`"0"`, `"1"`, …) with
    /// one entry per output, in either format:
    ///
    /// - `cardnames_onnx.json` (the fp32, fp16 and int8 ViTs): an object with the
    ///   passcode as a decimal string in `card_id`, such as
    ///   `{"EN": "Monster Reborn", "FR": "…", "card_id": "83764719", "label": "…"}`;
    /// - `card_labels_yugiscan.json` (the Yugiscan ViT): a plain `Name-<passcode>`
    ///   string, such as `"Monster-Reborn-83764719"`.
    ///
    /// # Errors
    ///
    /// [`LabelsError`] if the JSON is not in one of those formats, an index is missing,
    /// or a passcode is not an integer.
    pub(crate) fn from_json(json: &str) -> Result<Self, LabelsError> {
        let entries: HashMap<String, Entry> = serde_json::from_str(json)?;
        let mut by_index = Vec::with_capacity(entries.len());
        for (key, entry) in entries {
            let index: usize = key.parse().map_err(|_| LabelsError::Key(key))?;
            by_index.push((index, entry));
        }
        by_index.sort_unstable_by_key(|&(index, _)| index);
        let passcodes = by_index
            .into_iter()
            .enumerate()
            .map(|(expected, (index, entry))| {
                if index != expected {
                    // Keys are unique and sorted, so the first mismatch is a gap.
                    return Err(LabelsError::MissingIndex(expected));
                }
                entry.passcode().ok_or_else(|| LabelsError::Passcode {
                    index,
                    entry: entry.into_text(),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { passcodes })
    }

    /// The passcode of output `index`, or `None` past the last output.
    #[must_use]
    pub(crate) fn passcode(&self, index: usize) -> Option<Passcode> {
        self.passcodes.get(index).copied()
    }

    /// How many outputs the labels cover.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.passcodes.len()
    }
}

/// One labels-file entry, in either format.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Entry {
    /// `card_labels_yugiscan.json`: `Name-<passcode>`.
    Plain(String),
    /// `cardnames_onnx.json`: names by language and more, with the passcode in
    /// `card_id`.
    Card { card_id: String },
}

impl Entry {
    fn passcode(&self) -> Option<Passcode> {
        let digits = match self {
            Self::Plain(label) => label.rsplit_once('-')?.1,
            Self::Card { card_id } => card_id,
        };
        // `u32::from_str` accepts a leading `+`; a passcode is digits only.
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        digits.parse().ok().map(Passcode)
    }

    fn into_text(self) -> String {
        match self {
            Self::Plain(text) | Self::Card { card_id: text } => text,
        }
    }
}

/// Why a labels file could not be read.
#[derive(Debug, Error)]
pub(crate) enum LabelsError {
    /// The file is not a JSON object of label entries in a known format.
    #[error("labels are not in a known format: {0}")]
    Format(#[from] serde_json::Error),
    /// A key is not an output index.
    #[error("label key {0:?} is not an output index")]
    Key(String),
    /// No entry for an output index below the largest one.
    #[error("no label for output {0}")]
    MissingIndex(usize),
    /// An entry's passcode is not an integer.
    #[error("label for output {index} has no integer passcode: {entry:?}")]
    Passcode {
        /// The output index.
        index: usize,
        /// The passcode text (`card_id`) or the plain label it was read from.
        entry: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardnames_entries_map_each_output_to_its_card_id() {
        let json = r#"{
            "1": {"EN": "Monster Reborn", "FR": "Monster Reborn", "card_id": "83764719",
                  "label": "Monster-Reborn-83764719"},
            "0": {"EN": "\"A\" Cell Breeding Device", "FR": "Incubateur Cellule \"A\"",
                  "JA": "「A」細胞培養装置", "card_id": "34541863",
                  "label": "\"A\"-Cell-Breeding-Device-34541863"}
        }"#;

        let labels = Labels::from_json(json).unwrap();

        assert_eq!(labels.len(), 2);
        assert_eq!(labels.passcode(0), Some(Passcode(34_541_863)));
        assert_eq!(labels.passcode(1), Some(Passcode(83_764_719)));
        assert_eq!(labels.passcode(2), None);
    }

    #[test]
    fn yugiscan_entries_map_each_output_to_the_number_after_the_last_dash() {
        let json = r#"{
            "0": "\"A\"-Cell-Breeding-Device-34541863",
            "1": "Monster-Reborn-83764719",
            "2": "Leading-Zeros-00012345"
        }"#;

        let labels = Labels::from_json(json).unwrap();

        assert_eq!(labels.len(), 3);
        assert_eq!(labels.passcode(0), Some(Passcode(34_541_863)));
        assert_eq!(labels.passcode(1), Some(Passcode(83_764_719)));
        assert_eq!(labels.passcode(2), Some(Passcode(12_345)));
    }

    #[test]
    fn labels_with_a_gap_a_bad_key_or_a_non_integer_passcode_are_rejected() {
        let gap = r#"{"0": "A-1", "2": "C-3"}"#;
        let bad_key = r#"{"0": "A-1", "one": "B-2"}"#;
        let no_number = r#"{"0": "Monster-Reborn"}"#;
        let bad_card_id = r#"{"0": {"EN": "A", "card_id": "12a", "label": "A-12"}}"#;
        let not_an_object = r#"["A-1", "B-2"]"#;

        assert!(matches!(
            Labels::from_json(gap),
            Err(LabelsError::MissingIndex(1))
        ));
        assert!(matches!(Labels::from_json(bad_key), Err(LabelsError::Key(key)) if key == "one"));
        assert!(matches!(
            Labels::from_json(no_number),
            Err(LabelsError::Passcode { index: 0, .. })
        ));
        assert!(matches!(
            Labels::from_json(bad_card_id),
            Err(LabelsError::Passcode { index: 0, .. })
        ));
        assert!(matches!(
            Labels::from_json(not_an_object),
            Err(LabelsError::Format(_))
        ));
    }
}
