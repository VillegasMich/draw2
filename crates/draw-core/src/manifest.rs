//! The model files a version of draw-core expects, and where to get them.
//!
//! The values below are pinned to one commit of the Hugging Face repo. Change them only
//! together with code that expects different model files.

/// One of the interchangeable classifier builds, trading accuracy against size and speed.
///
/// Every variant uses the same detector. The default, [`ModelVariant::Fp16`], is the one
/// the DRAW 2 web demo preselects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModelVariant {
    /// Full-precision ViT classifier (largest, most accurate).
    Fp32,
    /// Half-precision ViT classifier.
    #[default]
    Fp16,
    /// 8-bit integer ViT classifier.
    Int8,
    /// 8-bit integer Yugiscan ViT classifier, with its own labels file.
    YugiscanInt8,
}

impl ModelVariant {
    /// Every variant, in the order listed above.
    pub const ALL: [Self; 4] = [Self::Fp32, Self::Fp16, Self::Int8, Self::YugiscanInt8];
}

/// What a model file is used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelRole {
    /// The YOLO card detector, which finds quads in a frame.
    Detector,
    /// The ViT classifier, which scores candidates for a card crop.
    Classifier,
    /// The JSON file mapping classifier outputs to passcodes.
    Labels,
}

/// One file to download from the model repo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelFile {
    /// What the file is used for.
    pub role: ModelRole,
    /// Path inside the model repo, such as `onnx/ygo_yolo.onnx`.
    pub path: &'static str,
    /// SHA-256 of the file's contents, as 64 lowercase hex characters.
    pub sha256: &'static str,
    /// File size, in bytes.
    pub size: u64,
}

/// The model files this version of draw-core requires, with where to get them and how to
/// verify them.
///
/// There is one manifest per crate version: [`ModelManifest::PINNED`]. Files are served by
/// Hugging Face at a fixed commit, so a download always matches its listed SHA-256 and
/// size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelManifest {
    repo: &'static str,
    commit: &'static str,
    fp32: [ModelFile; 3],
    fp16: [ModelFile; 3],
    int8: [ModelFile; 3],
    yugiscan_int8: [ModelFile; 3],
}

const DETECTOR: ModelFile = ModelFile {
    role: ModelRole::Detector,
    path: "onnx/ygo_yolo.onnx",
    sha256: "449e0d42cbf8429b170abf44d50cda29beeeb05dc80890102e12c06d6358cf97",
    size: 39_016_336,
};

const CARDNAMES_LABELS: ModelFile = ModelFile {
    role: ModelRole::Labels,
    path: "onnx/cardnames_onnx.json",
    sha256: "70d13d586eee498b33ee493b48cd564fbffcb99b0476897978de1fd7d856f639",
    size: 1_768_550,
};

const YUGISCAN_LABELS: ModelFile = ModelFile {
    role: ModelRole::Labels,
    path: "onnx/card_labels_yugiscan.json",
    sha256: "4f558eeef0be9f530d1f4b58eab0520af21ef845c6482105161be98db290b90b",
    size: 575_524,
};

const fn classifier(path: &'static str, sha256: &'static str, size: u64) -> ModelFile {
    ModelFile {
        role: ModelRole::Classifier,
        path,
        sha256,
        size,
    }
}

impl ModelManifest {
    /// The manifest for this version of draw-core.
    pub const PINNED: Self = Self {
        repo: "HichTala/draw2",
        commit: "7a690eecf6aef5c812ba9a5a1d3e1ced4968b5b3",
        fp32: [
            DETECTOR,
            classifier(
                "onnx/vit_fp32.onnx",
                "8e6b63c64e081f6f5326adf0c758247bb88764224c2380d547e9c9005e574ca4",
                386_354_475,
            ),
            CARDNAMES_LABELS,
        ],
        fp16: [
            DETECTOR,
            classifier(
                "onnx/vit_fp16.onnx",
                "ca67e39f26d65964feff81072952402462ac636cf457eab6afd75b0c4ef17bc8",
                193_077_924,
            ),
            CARDNAMES_LABELS,
        ],
        int8: [
            DETECTOR,
            classifier(
                "onnx/vit_int8.onnx",
                "f2da9ce6b2c02498e0c9f8fc0b7817eeebd88bfe8491c8c7f83055dfd57e5b4a",
                97_771_550,
            ),
            CARDNAMES_LABELS,
        ],
        yugiscan_int8: [
            DETECTOR,
            classifier(
                "onnx/vit_yugiscan_int8.onnx",
                "bcde9c36204d391f44d11038a5dcd7742b99f1fcbdf26583670be29010cd8bd9",
                97_580_672,
            ),
            YUGISCAN_LABELS,
        ],
    };

    /// The Hugging Face model repo, as `owner/name`.
    #[must_use]
    pub fn repo(&self) -> &'static str {
        self.repo
    }

    /// The pinned commit of [`repo`](Self::repo), as a full 40-character hex SHA-1.
    #[must_use]
    pub fn commit(&self) -> &'static str {
        self.commit
    }

    /// The files `variant` needs: one detector, one classifier and one labels file.
    #[must_use]
    pub fn files(&self, variant: ModelVariant) -> &[ModelFile] {
        match variant {
            ModelVariant::Fp32 => &self.fp32,
            ModelVariant::Fp16 => &self.fp16,
            ModelVariant::Int8 => &self.int8,
            ModelVariant::YugiscanInt8 => &self.yugiscan_int8,
        }
    }

    /// The download URL for `file`, at the pinned commit.
    #[must_use]
    pub fn url(&self, file: &ModelFile) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.commit, file.path
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: ModelManifest = ModelManifest::PINNED;

    fn is_lower_hex(s: &str, len: usize) -> bool {
        s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    }

    fn paths_for(variant: ModelVariant, role: ModelRole) -> Vec<&'static str> {
        MANIFEST
            .files(variant)
            .iter()
            .filter(|file| file.role == role)
            .map(|file| file.path)
            .collect()
    }

    #[test]
    fn every_variant_has_one_detector_one_classifier_and_one_labels_file() {
        for variant in ModelVariant::ALL {
            assert_eq!(MANIFEST.files(variant).len(), 3, "{variant:?}");
            for role in [
                ModelRole::Detector,
                ModelRole::Classifier,
                ModelRole::Labels,
            ] {
                assert_eq!(paths_for(variant, role).len(), 1, "{variant:?} {role:?}");
            }
        }
    }

    #[test]
    fn variants_use_the_shared_detector_and_their_own_classifier_and_labels() {
        use ModelRole::{Classifier, Detector, Labels};
        use ModelVariant::{Fp16, Fp32, Int8, YugiscanInt8};

        let expected = [
            (Fp32, "onnx/vit_fp32.onnx", "onnx/cardnames_onnx.json"),
            (Fp16, "onnx/vit_fp16.onnx", "onnx/cardnames_onnx.json"),
            (Int8, "onnx/vit_int8.onnx", "onnx/cardnames_onnx.json"),
            (
                YugiscanInt8,
                "onnx/vit_yugiscan_int8.onnx",
                "onnx/card_labels_yugiscan.json",
            ),
        ];
        for (variant, classifier, labels) in expected {
            assert_eq!(paths_for(variant, Detector), ["onnx/ygo_yolo.onnx"]);
            assert_eq!(paths_for(variant, Classifier), [classifier], "{variant:?}");
            assert_eq!(paths_for(variant, Labels), [labels], "{variant:?}");
        }
    }

    #[test]
    fn every_file_has_a_lowercase_hex_sha256_and_a_nonzero_size() {
        for variant in ModelVariant::ALL {
            for file in MANIFEST.files(variant) {
                assert!(is_lower_hex(file.sha256, 64), "{}", file.path);
                assert!(file.size > 0, "{}", file.path);
            }
        }
    }

    #[test]
    fn the_manifest_pins_hichtala_draw2_at_a_full_40_hex_commit() {
        assert_eq!(MANIFEST.repo(), "HichTala/draw2");
        assert!(is_lower_hex(MANIFEST.commit(), 40), "{}", MANIFEST.commit());
    }

    #[test]
    fn download_urls_embed_the_pinned_commit_not_main() {
        for variant in ModelVariant::ALL {
            for file in MANIFEST.files(variant) {
                let url = MANIFEST.url(file);

                assert_eq!(
                    url,
                    format!(
                        "https://huggingface.co/HichTala/draw2/resolve/{}/{}",
                        MANIFEST.commit(),
                        file.path
                    )
                );
                assert!(!url.contains("/main/"), "{url}");
            }
        }
    }
}
