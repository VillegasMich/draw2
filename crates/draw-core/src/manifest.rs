//! The model files a version of draw-core expects, and where to get them.

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
