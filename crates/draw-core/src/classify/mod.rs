//! Classification: scores the candidates for a card crop with a ViT classifier, settles
//! the card's orientation, and decides the recognition.
//!
//! A port of `correctRotation`, `classifyWithRotationFallback`, `preprocessViT` and
//! `topK` from draw2's browser pipeline (`docs/scripts/pipeline.js`), and of the deck
//! filter of the Python pipeline (`src/draw/draw.py`).

pub(crate) mod orientation;
