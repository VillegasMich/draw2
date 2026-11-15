//! The ONNX Runtime environment shared by every model session in the process.
//!
//! `ort` allows only **one environment per process**: a second
//! `ort::init().build()` fails with "only one environment is allowed per process", even
//! after the first is dropped. Every session needs one, so every stage gets it from
//! [`environment`]: the detector (`detect`) and the classifier (ticket A2) alike. Never
//! call `ort::init()` anywhere else.
//!
//! Ticket A3 owns this module once it wires the stages into the `Recognizer` (and adds
//! the execution providers here, if they are configured per environment).

use std::sync::{Mutex, PoisonError};

use ort::environment::Environment;

/// The process's ONNX Runtime environment, created on first use and shared after that.
///
/// # Errors
///
/// The [`ort::Error`] from creating the environment, if this is the first call and
/// ONNX Runtime cannot start. A later call tries again.
pub(crate) fn environment() -> Result<Environment, ort::Error> {
    static ENVIRONMENT: Mutex<Option<Environment>> = Mutex::new(None);
    // The guarded value is only ever replaced whole, so a poisoned lock still holds a
    // consistent value.
    let mut shared = ENVIRONMENT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(environment) = shared.as_ref() {
        return Ok(environment.clone());
    }
    let environment = ort::init().with_name("draw-core").build()?;
    *shared = Some(environment.clone());
    Ok(environment)
}
