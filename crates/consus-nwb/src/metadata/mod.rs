//! NWB session metadata model.
//!
//! Canonical Rust types for the required NWB 2.x session-level metadata
//! stored as scalar string attributes on the root HDF5 group.
//!
//! ## Specification
//!
//! NWB 2.x requires the following attributes on the root group:
//!
//! | Attribute              | Type   | Description                                  |
//! |------------------------|--------|----------------------------------------------|
//! | `identifier`           | string | Globally unique session identifier           |
//! | `session_description`  | string | Human-readable description of the session    |
//! | `session_start_time`   | string | ISO 8601 timestamp of the session start time |
//!
//! Optional attributes (`experimenter`, `institution`, `lab`,
//! `related_publications`, `keywords`, `notes`, `pharmacology`,
//! `protocol`, `slices`, `source_script`, `surgery`, `virus`,
//! `timestamps_reference_frame`) are not modelled here; they are
//! roadmap items for later milestones.
//!
//! ## Invariants
//!
//! - All three required fields are non-empty strings after construction.
//! - The model is independent of wire encoding and HDF5 layout details.

#[cfg(feature = "alloc")]
use alloc::string::String;

/// Required NWB 2.x session-level metadata.
///
/// Extracted from the scalar string attributes on the root HDF5 group of an
/// NWB file.  All three fields are required by the NWB 2.x specification;
/// files missing any of them fail validation before this struct is produced.
///
/// ## Example
///
/// ```
/// # #[cfg(feature = "alloc")] {
/// use consus_nwb::metadata::NwbSessionMetadata;
///
/// let meta = NwbSessionMetadata::new(
///     "sub-001_ses-20230101",
///     "Freely moving mouse on linear track",
///     "2023-01-01T09:00:00+00:00",
/// );
/// assert_eq!(meta.identifier(), "sub-001_ses-20230101");
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NwbSessionMetadata {
    /// Globally unique session identifier (maps to `identifier` attribute).
    #[cfg(feature = "alloc")]
    identifier: String,

    /// Human-readable description of the session (maps to `session_description`).
    #[cfg(feature = "alloc")]
    session_description: String,

    /// ISO 8601 session start timestamp (maps to `session_start_time`).
    #[cfg(feature = "alloc")]
    session_start_time: String,
}

#[cfg(feature = "alloc")]
impl NwbSessionMetadata {
    /// Construct a session metadata record from the three required fields.
    ///
    /// ## Arguments
    ///
    /// - `identifier`          — globally unique session identifier
    /// - `session_description` — human-readable session description
    /// - `session_start_time`  — ISO 8601 session start time string
    #[must_use]
    pub fn new(
        identifier: impl Into<String>,
        session_description: impl Into<String>,
        session_start_time: impl Into<String>,
    ) -> Self {
        Self {
            identifier: identifier.into(),
            session_description: session_description.into(),
            session_start_time: session_start_time.into(),
        }
    }

    /// Return the globally unique session identifier.
    #[must_use]
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    /// Return the human-readable session description.
    #[must_use]
    pub fn session_description(&self) -> &str {
        &self.session_description
    }

    /// Return the ISO 8601 session start time string.
    #[must_use]
    pub fn session_start_time(&self) -> &str {
        &self.session_start_time
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    // ── NwbSessionMetadata::new and accessors ──────────────────────────────

    #[test]
    fn new_stores_all_three_fields() {
        // Theorem: new(id, desc, ts) stores all three values accessibly.
        let meta = NwbSessionMetadata::new(
            "sub-001_ses-20230101",
            "Freely moving mouse on linear track",
            "2023-01-01T09:00:00+00:00",
        );

        assert_eq!(meta.identifier(), "sub-001_ses-20230101");
        assert_eq!(
            meta.session_description(),
            "Freely moving mouse on linear track"
        );
        assert_eq!(meta.session_start_time(), "2023-01-01T09:00:00+00:00");
    }

    #[test]
    fn new_accepts_string_owned() {
        // Theorem: new accepts owned String values as well as &str.
        let id = String::from("session-42");
        let desc = String::from("Test session");
        let ts = String::from("2024-06-15T12:00:00+00:00");

        let meta = NwbSessionMetadata::new(id.clone(), desc.clone(), ts.clone());

        assert_eq!(meta.identifier(), id.as_str());
        assert_eq!(meta.session_description(), desc.as_str());
        assert_eq!(meta.session_start_time(), ts.as_str());
    }

    #[test]
    fn equality_holds_for_identical_values() {
        // Theorem: two structs with identical fields compare equal.
        let a = NwbSessionMetadata::new("id", "desc", "2023-01-01T00:00:00+00:00");
        let b = NwbSessionMetadata::new("id", "desc", "2023-01-01T00:00:00+00:00");
        assert_eq!(a, b);
    }

    #[test]
    fn equality_fails_on_differing_identifier() {
        let a = NwbSessionMetadata::new("id-a", "desc", "2023-01-01T00:00:00+00:00");
        let b = NwbSessionMetadata::new("id-b", "desc", "2023-01-01T00:00:00+00:00");
        assert_ne!(a, b);
    }

    #[test]
    fn equality_fails_on_differing_description() {
        let a = NwbSessionMetadata::new("id", "desc-a", "2023-01-01T00:00:00+00:00");
        let b = NwbSessionMetadata::new("id", "desc-b", "2023-01-01T00:00:00+00:00");
        assert_ne!(a, b);
    }

    #[test]
    fn equality_fails_on_differing_start_time() {
        let a = NwbSessionMetadata::new("id", "desc", "2023-01-01T00:00:00+00:00");
        let b = NwbSessionMetadata::new("id", "desc", "2024-06-01T00:00:00+00:00");
        assert_ne!(a, b);
    }

    #[test]
    fn clone_produces_equal_independent_copy() {
        // Theorem: clone yields a value equal to the original and independently owned.
        let original = NwbSessionMetadata::new(
            "clone-test-id",
            "Clone test session",
            "2023-03-15T08:30:00+00:00",
        );
        let cloned = original.clone();

        assert_eq!(original, cloned);

        // Verify independence by comparing field values.
        assert_eq!(cloned.identifier(), original.identifier());
        assert_eq!(cloned.session_description(), original.session_description());
        assert_eq!(cloned.session_start_time(), original.session_start_time());
    }

    #[test]
    fn debug_output_contains_field_values() {
        // Theorem: Debug output contains the identifier string.
        let meta = NwbSessionMetadata::new(
            "debug-session-id",
            "Debug test session",
            "2023-01-01T00:00:00+00:00",
        );
        let debug_str = alloc::format!("{:?}", meta);
        assert!(
            debug_str.contains("debug-session-id"),
            "Debug output should contain the identifier: {debug_str}"
        );
    }

    #[test]
    fn empty_string_fields_are_stored_verbatim() {
        // Theorem: empty strings are stored without modification.
        // NWB validation (not this struct) enforces non-empty constraints.
        let meta = NwbSessionMetadata::new("", "", "");
        assert_eq!(meta.identifier(), "");
        assert_eq!(meta.session_description(), "");
        assert_eq!(meta.session_start_time(), "");
    }

    #[test]
    fn unicode_fields_are_preserved() {
        // Theorem: Unicode content in all three fields is stored and retrieved
        // without modification.
        let meta = NwbSessionMetadata::new(
            "日本語-session-001",
            "実験の説明: マウスの自由移動",
            "2023-07-04T00:00:00+09:00",
        );
        assert_eq!(meta.identifier(), "日本語-session-001");
        assert_eq!(meta.session_description(), "実験の説明: マウスの自由移動");
        assert_eq!(meta.session_start_time(), "2023-07-04T00:00:00+09:00");
    }
}

// ---------------------------------------------------------------------------
// NwbSubjectMetadata
// ---------------------------------------------------------------------------

/// Optional NWB 2.x subject-level metadata.
///
/// Extracted from the `general/subject` HDF5 group of an NWB file.
/// All fields are optional; the NWB 2.x specification does not require any
/// particular attribute within the Subject group.
///
/// ## Specification
///
/// NWB 2.x Subject group (`general/subject`) stores these optional string
/// attributes:
///
/// | HDF5 attribute | Field         | Example              |
/// |----------------|---------------|----------------------|
/// | `subject_id`   | `subject_id`  | `"sub-001"`          |
/// | `species`      | `species`     | `"Mus musculus"`     |
/// | `sex`          | `sex`         | `"M"`, `"F"`, `"U"`, `"O"` |
/// | `age`          | `age`         | `"P90D"` (ISO 8601)  |
/// | `description`  | `description` | `"C57BL/6J mouse"`   |
///
/// Reference: <https://nwb-schema.readthedocs.io/en/latest/format_description.html>
///
/// ## Example
///
/// ```
/// # #[cfg(feature = "alloc")] {
/// use consus_nwb::metadata::NwbSubjectMetadata;
///
/// let subj = NwbSubjectMetadata::from_parts(
///     Some("sub-001".into()),
///     Some("Mus musculus".into()),
///     Some("M".into()),
///     Some("P90D".into()),
///     Some("C57BL/6J mouse".into()),
/// );
/// assert_eq!(subj.subject_id(), Some("sub-001"));
/// assert_eq!(subj.species(), Some("Mus musculus"));
/// # }
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NwbSubjectMetadata {
    /// Unique subject identifier (maps to `subject_id` attribute).
    subject_id: Option<String>,

    /// Binomial species name (maps to `species` attribute).
    species: Option<String>,

    /// Biological sex code: `"M"`, `"F"`, `"U"` (unknown), or `"O"` (other).
    sex: Option<String>,

    /// Subject age in ISO 8601 duration format, e.g. `"P90D"` (maps to `age`).
    age: Option<String>,

    /// Free-text description of the subject (maps to `description` attribute).
    description: Option<String>,
}

#[cfg(feature = "alloc")]
impl NwbSubjectMetadata {
    /// Construct from constituent optional fields.
    ///
    /// All fields are optional; pass `None` for absent metadata.
    ///
    /// ## Arguments
    ///
    /// - `subject_id`  — unique subject identifier (e.g. `"sub-001"`)
    /// - `species`     — binomial species name (e.g. `"Mus musculus"`)
    /// - `sex`         — sex code: `"M"`, `"F"`, `"U"`, or `"O"`
    /// - `age`         — ISO 8601 duration string (e.g. `"P90D"`)
    /// - `description` — free-text subject description
    #[must_use]
    pub fn from_parts(
        subject_id: Option<String>,
        species: Option<String>,
        sex: Option<String>,
        age: Option<String>,
        description: Option<String>,
    ) -> Self {
        Self {
            subject_id,
            species,
            sex,
            age,
            description,
        }
    }

    /// Return the subject identifier, or `None` if absent.
    #[must_use]
    pub fn subject_id(&self) -> Option<&str> {
        self.subject_id.as_deref()
    }

    /// Return the binomial species name, or `None` if absent.
    #[must_use]
    pub fn species(&self) -> Option<&str> {
        self.species.as_deref()
    }

    /// Return the sex code (`"M"`, `"F"`, `"U"`, or `"O"`), or `None` if absent.
    #[must_use]
    pub fn sex(&self) -> Option<&str> {
        self.sex.as_deref()
    }

    /// Return the ISO 8601 age duration string, or `None` if absent.
    #[must_use]
    pub fn age(&self) -> Option<&str> {
        self.age.as_deref()
    }

    /// Return the free-text subject description, or `None` if absent.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

#[cfg(all(test, feature = "alloc"))]
mod subject_tests {
    use super::*;

    // ── NwbSubjectMetadata::from_parts and accessors ───────────────────────

    #[test]
    fn from_parts_all_some_stores_all_fields() {
        // Theorem: from_parts with all Some values stores each field accessibly.
        // Values follow NWB specimen conventions for a C57BL/6J male mouse.
        let subj = NwbSubjectMetadata::from_parts(
            Some("sub-001".into()),
            Some("Mus musculus".into()),
            Some("M".into()),
            Some("P90D".into()),
            Some("C57BL/6J mouse".into()),
        );

        assert_eq!(subj.subject_id(), Some("sub-001"));
        assert_eq!(subj.species(), Some("Mus musculus"));
        assert_eq!(subj.sex(), Some("M"));
        assert_eq!(subj.age(), Some("P90D"));
        assert_eq!(subj.description(), Some("C57BL/6J mouse"));
    }

    #[test]
    fn from_parts_all_none_returns_all_none() {
        // Theorem: from_parts with all None values returns None for every accessor.
        let subj = NwbSubjectMetadata::from_parts(None, None, None, None, None);

        assert_eq!(subj.subject_id(), None);
        assert_eq!(subj.species(), None);
        assert_eq!(subj.sex(), None);
        assert_eq!(subj.age(), None);
        assert_eq!(subj.description(), None);
    }

    #[test]
    fn from_parts_partial_some() {
        // Theorem: from_parts with only species and age set returns None for
        // the other three fields and Some for species and age.
        // Species and age follow NWB specimen conventions for a rat subject.
        let subj = NwbSubjectMetadata::from_parts(
            None,
            Some("Rattus norvegicus".into()),
            None,
            Some("P120D".into()),
            None,
        );

        assert_eq!(subj.subject_id(), None);
        assert_eq!(subj.species(), Some("Rattus norvegicus"));
        assert_eq!(subj.sex(), None);
        assert_eq!(subj.age(), Some("P120D"));
        assert_eq!(subj.description(), None);
    }

    #[test]
    fn equality_holds_for_identical_values() {
        // Theorem: two structs constructed with identical arguments compare equal.
        let a = NwbSubjectMetadata::from_parts(
            Some("sub-002".into()),
            Some("Mus musculus".into()),
            Some("F".into()),
            Some("P60D".into()),
            Some("C57BL/6J female mouse".into()),
        );
        let b = NwbSubjectMetadata::from_parts(
            Some("sub-002".into()),
            Some("Mus musculus".into()),
            Some("F".into()),
            Some("P60D".into()),
            Some("C57BL/6J female mouse".into()),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn equality_fails_on_differing_species() {
        // Theorem: two structs differing only in species compare not equal.
        let a = NwbSubjectMetadata::from_parts(
            Some("sub-003".into()),
            Some("Mus musculus".into()),
            Some("M".into()),
            Some("P90D".into()),
            None,
        );
        let b = NwbSubjectMetadata::from_parts(
            Some("sub-003".into()),
            Some("Rattus norvegicus".into()),
            Some("M".into()),
            Some("P90D".into()),
            None,
        );
        assert_ne!(a, b);
    }

    #[test]
    fn clone_produces_equal_independent_copy() {
        // Theorem: clone yields a value equal to the original and independently owned.
        let original = NwbSubjectMetadata::from_parts(
            Some("sub-004".into()),
            Some("Mus musculus".into()),
            Some("U".into()),
            Some("P45D".into()),
            Some("Transgenic line: Thy1-GCaMP6f".into()),
        );
        let cloned = original.clone();

        assert_eq!(original, cloned);
        assert_eq!(cloned.subject_id(), original.subject_id());
        assert_eq!(cloned.species(), original.species());
        assert_eq!(cloned.sex(), original.sex());
        assert_eq!(cloned.age(), original.age());
        assert_eq!(cloned.description(), original.description());
    }

    #[test]
    fn debug_output_contains_field_value() {
        // Theorem: Debug output contains the subject_id string.
        let subj =
            NwbSubjectMetadata::from_parts(Some("sub-debug-001".into()), None, None, None, None);
        let debug_str = alloc::format!("{:?}", subj);
        assert!(
            debug_str.contains("sub-debug-001"),
            "Debug output should contain subject_id: {debug_str}"
        );
    }
}
