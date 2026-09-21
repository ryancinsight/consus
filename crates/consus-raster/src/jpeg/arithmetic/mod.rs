//! Bounded JPEG QM arithmetic decision decoding.
//!
//! The register layout, conditional exchange, byte input, and renormalization
//! follow ITU-T T.81 Annex D, sections D.2.3 through D.2.7. The state table is
//! Table D.3. The independently maintained decoder in libjpeg-turbo 3.1.2
//! (`src/jdarith.c` and `src/jaricom.c`) is the differential implementation.
//! See <https://www.w3.org/Graphics/JPEG/itu-t81.pdf> and
//! <https://skia.googlesource.com/external/github.com/libjpeg-turbo/libjpeg-turbo.git/+/refs/tags/3.1.2/src/jdarith.c>.

mod reader;
mod state;

pub(super) use reader::Reader;
pub(super) use state::State;
