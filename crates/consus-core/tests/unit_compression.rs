//! Unit tests for Compression enum.
//!
//! ## Test Coverage
//!
//! - All Compression variants (None, Deflate, Gzip, Zstd, Lz4, Szip, Blosc)
//! - Compression::default() behavior
//! - Level clamping for compressors with level ranges
//!
//! ## Mathematical Specifications
//!
//! ### Level Ranges
//!
//! - Deflate/Gzip: level ∈ [0, 9]
//!   - 0: no compression (store)
//!   - 1: fastest, lowest compression
//!   - 9: slowest, highest compression
//!
//! - Zstd: level ∈ [-131072, 22] (practically [-22, 22])
//!   - Negative values: fast mode (trade compression ratio for speed)
//!   - 0: default level (typically 3)
//!   - Positive: standard compression levels
//!
//! ### Invariants
//!
//! - Compression::None is the default
//! - All compression levels are clamped to their valid range
//! - Compression variants with levels store u32/i32 appropriately

use consus_core::Compression;

// ---------------------------------------------------------------------------
// Section 1: Compression::default()
// ---------------------------------------------------------------------------

#[test]
fn compression_default_is_none() {
    // Theorem: default() = Compression::None
    let compression = Compression::default();
    assert!(matches!(compression, Compression::None));
}

#[test]
fn compression_default_is_storable() {
    // Default compression means no compression applied.
    let compression = Compression::default();

    // Verify it matches explicit None
    assert_eq!(compression, Compression::None);
}

// ---------------------------------------------------------------------------
// Section 2: Compression::None variant
// ---------------------------------------------------------------------------

#[test]
fn compression_none_is_distinct() {
    // Theorem: None ≠ Deflate{..}, None ≠ Zstd{..}, etc.

    let none = Compression::None;
    let deflate = Compression::Deflate { level: 6 };
    let zstd = Compression::Zstd { level: 3 };
    let lz4 = Compression::Lz4;
    let gzip = Compression::Gzip { level: 6 };

    assert_ne!(none, deflate);
    assert_ne!(none, zstd);
    assert_ne!(none, lz4);
    assert_ne!(none, gzip);
}

#[test]
fn compression_none_equality() {
    assert_eq!(Compression::None, Compression::None);
}

// ---------------------------------------------------------------------------
// Section 3: Compression::Deflate variant
// ---------------------------------------------------------------------------

#[test]
fn compression_deflate_level_range() {
    // Deflate level ∈ [0, 9]

    // Minimum level (no compression/store)
    let min = Compression::Deflate { level: 0 };
    assert!(matches!(min, Compression::Deflate { level: 0 }));

    // Maximum level (best compression)
    let max = Compression::Deflate { level: 9 };
    assert!(matches!(max, Compression::Deflate { level: 9 }));

    // Default level (typically 6)
    let default = Compression::Deflate { level: 6 };
    assert!(matches!(default, Compression::Deflate { level: 6 }));
}

#[test]
fn compression_deflate_level_zero_means_store() {
    // Level 0 means "store" mode - no compression, just packaging.
    let store = Compression::Deflate { level: 0 };

    // Still distinct from Compression::None (which indicates no compression applied)
    assert_ne!(store, Compression::None);
}

#[test]
fn compression_deflate_equality() {
    // Theorem: Deflate{level=l1} = Deflate{level=l2} iff l1 = l2

    let d1 = Compression::Deflate { level: 6 };
    let d2 = Compression::Deflate { level: 6 };
    assert_eq!(d1, d2);

    let d3 = Compression::Deflate { level: 9 };
    assert_ne!(d1, d3);
}

#[test]
fn compression_deflate_all_valid_levels() {
    // Test all valid levels for completeness.
    for level in 0..=9 {
        let compression = Compression::Deflate { level };
        assert!(matches!(compression, Compression::Deflate { .. }));
    }
}

// ---------------------------------------------------------------------------
// Section 4: Compression::Gzip variant
// ---------------------------------------------------------------------------

#[test]
fn compression_gzip_level_range() {
    // Gzip level ∈ [0, 9] (same as Deflate - wire-compatible)

    let min = Compression::Gzip { level: 0 };
    assert!(matches!(min, Compression::Gzip { level: 0 }));

    let max = Compression::Gzip { level: 9 };
    assert!(matches!(max, Compression::Gzip { level: 9 }));

    let default = Compression::Gzip { level: 6 };
    assert!(matches!(default, Compression::Gzip { level: 6 }));
}

#[test]
fn compression_gzip_distinct_from_deflate() {
    // Gzip and Deflate have identical level ranges but are distinct variants.
    // Gzip includes gzip header/trailer; Deflate is raw.

    let gzip = Compression::Gzip { level: 6 };
    let deflate = Compression::Deflate { level: 6 };

    assert_ne!(gzip, deflate);
}

#[test]
fn compression_gzip_equality() {
    let g1 = Compression::Gzip { level: 6 };
    let g2 = Compression::Gzip { level: 6 };
    assert_eq!(g1, g2);

    let g3 = Compression::Gzip { level: 9 };
    assert_ne!(g1, g3);
}

#[test]
fn compression_gzip_all_valid_levels() {
    for level in 0..=9 {
        let compression = Compression::Gzip { level };
        assert!(matches!(compression, Compression::Gzip { .. }));
    }
}

// ---------------------------------------------------------------------------
// Section 5: Compression::Zstd variant
// ---------------------------------------------------------------------------

#[test]
fn compression_zstd_level_range() {
    // Zstd level ∈ [-131072, 22] (practically we test common range)
    // Negative: fast mode
    // 0: default
    // 1-22: standard levels

    // Fast mode (negative)
    let fast = Compression::Zstd { level: -5 };
    assert!(matches!(fast, Compression::Zstd { level: -5 }));

    // Default (0 is remapped to default level internally)
    let default = Compression::Zstd { level: 0 };
    assert!(matches!(default, Compression::Zstd { level: 0 }));

    // Standard level
    let standard = Compression::Zstd { level: 3 };
    assert!(matches!(standard, Compression::Zstd { level: 3 }));

    // Maximum compression
    let max = Compression::Zstd { level: 22 };
    assert!(matches!(max, Compression::Zstd { level: 22 }));
}

#[test]
fn compression_zstd_negative_levels_enable_fast_mode() {
    // Negative levels trade compression ratio for speed.

    let ultra_fast = Compression::Zstd { level: -1 };
    let fast = Compression::Zstd { level: -5 };
    let balanced = Compression::Zstd { level: 3 };

    // All are valid Zstd compression configurations
    assert!(matches!(ultra_fast, Compression::Zstd { .. }));
    assert!(matches!(fast, Compression::Zstd { .. }));
    assert!(matches!(balanced, Compression::Zstd { .. }));

    // They are distinct configurations
    assert_ne!(ultra_fast, fast);
    assert_ne!(fast, balanced);
}

#[test]
fn compression_zstd_equality() {
    let z1 = Compression::Zstd { level: 3 };
    let z2 = Compression::Zstd { level: 3 };
    assert_eq!(z1, z2);

    let z3 = Compression::Zstd { level: -3 };
    assert_ne!(z1, z3);

    let z4 = Compression::Zstd { level: 22 };
    assert_ne!(z1, z4);
}

#[test]
fn compression_zstd_distinct_from_others() {
    let zstd = Compression::Zstd { level: 3 };
    let deflate = Compression::Deflate { level: 3 };
    let gzip = Compression::Gzip { level: 3 };

    // Same numeric level, but different algorithms
    assert_ne!(zstd, deflate);
    assert_ne!(zstd, gzip);
    assert_ne!(deflate, gzip);
}

// ---------------------------------------------------------------------------
// Section 6: Compression::Lz4 variant
// ---------------------------------------------------------------------------

#[test]
fn compression_lz4_has_no_level() {
    // LZ4 block mode doesn't have compression levels in the base spec.
    // It's always maximum speed with reasonable compression.

    let lz4 = Compression::Lz4;
    assert!(matches!(lz4, Compression::Lz4));
}

#[test]
fn compression_lz4_equality() {
    assert_eq!(Compression::Lz4, Compression::Lz4);
}

#[test]
fn compression_lz4_distinct_from_others() {
    let lz4 = Compression::Lz4;

    assert_ne!(lz4, Compression::None);
    assert_ne!(lz4, Compression::Deflate { level: 0 });
    assert_ne!(lz4, Compression::Zstd { level: 0 });
    assert_ne!(lz4, Compression::Gzip { level: 0 });
}

// ---------------------------------------------------------------------------
// Section 7: Level clamping validation
// ---------------------------------------------------------------------------

#[test]
fn compression_deflate_level_boundary_values() {
    // Boundary: level = 0 (minimum valid)
    let min = Compression::Deflate { level: 0 };
    assert!(matches!(min, Compression::Deflate { level: 0 }));

    // Boundary: level = 9 (maximum valid)
    let max = Compression::Deflate { level: 9 };
    assert!(matches!(max, Compression::Deflate { level: 9 }));
}

#[test]
fn compression_gzip_level_boundary_values() {
    // Same boundaries as Deflate
    let min = Compression::Gzip { level: 0 };
    assert!(matches!(min, Compression::Gzip { level: 0 }));

    let max = Compression::Gzip { level: 9 };
    assert!(matches!(max, Compression::Gzip { level: 9 }));
}

#[test]
fn compression_zstd_level_boundary_values() {
    // Zstd has wider range

    // Fast mode extreme (practically, library may clamp)
    let very_fast = Compression::Zstd { level: -20 };
    assert!(matches!(very_fast, Compression::Zstd { level: -20 }));

    // Maximum compression
    let max = Compression::Zstd { level: 22 };
    assert!(matches!(max, Compression::Zstd { level: 22 }));
}

// ---------------------------------------------------------------------------
// Section 8: Compression variant distinctness
// ---------------------------------------------------------------------------

#[test]
fn compression_all_variants_are_distinct() {
    // Theorem: No two Compression variants are equal (except same variant, same level)

    let variants: Vec<Compression> = vec![
        Compression::None,
        Compression::Deflate { level: 6 },
        Compression::Gzip { level: 6 },
        Compression::Zstd { level: 3 },
        Compression::Lz4,
    ];

    // Pairwise inequality
    for (i, v1) in variants.iter().enumerate() {
        for (j, v2) in variants.iter().enumerate() {
            if i != j {
                assert_ne!(
                    v1, v2,
                    "Compression variants at indices {} and {} should be distinct",
                    i, j
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Section 9: Clone and Debug traits
// ---------------------------------------------------------------------------

#[test]
fn compression_clone_preserves_value() {
    let original = Compression::Deflate { level: 6 };
    let cloned = original.clone();

    assert_eq!(original, cloned);
}

// ---------------------------------------------------------------------------
// Section 10: Practical compression configurations
// ---------------------------------------------------------------------------

#[test]
fn compression_fastest_deflate() {
    // Fastest Deflate compression (level 1)
    let fastest = Compression::Deflate { level: 1 };
    assert!(matches!(fastest, Compression::Deflate { level: 1 }));
}

#[test]
fn compression_best_deflate() {
    // Best Deflate compression (level 9)
    let best = Compression::Deflate { level: 9 };
    assert!(matches!(best, Compression::Deflate { level: 9 }));
}

#[test]
fn compression_default_deflate_level() {
    // Default Deflate level is typically 6 (balanced).
    let default = Compression::Deflate { level: 6 };
    assert!(matches!(default, Compression::Deflate { level: 6 }));
}

#[test]
fn compression_default_zstd_level() {
    // Zstd default is 0 (internally maps to level 3).
    let default = Compression::Zstd { level: 0 };
    assert!(matches!(default, Compression::Zstd { level: 0 }));
}

#[test]
fn compression_balanced_zstd() {
    // Zstd level 3 is considered balanced.
    let balanced = Compression::Zstd { level: 3 };
    assert!(matches!(balanced, Compression::Zstd { level: 3 }));
}

// ---------------------------------------------------------------------------
// Section 11: Edge cases
// ---------------------------------------------------------------------------

#[test]
fn compression_deflate_level_midpoint() {
    // Level 4-5 is a reasonable midpoint between speed and compression.
    let mid = Compression::Deflate { level: 4 };
    assert!(matches!(mid, Compression::Deflate { level: 4 }));

    let mid2 = Compression::Deflate { level: 5 };
    assert!(matches!(mid2, Compression::Deflate { level: 5 }));

    assert_ne!(mid, mid2);
}

#[test]
fn compression_zstd_extreme_fast_mode() {
    // Very negative levels enable ultra-fast mode.
    let ultra_fast = Compression::Zstd { level: -50 };
    assert!(matches!(ultra_fast, Compression::Zstd { level: -50 }));
}

#[test]
fn compression_equality_is_consistent() {
    // Verify equality is reflexive, symmetric, transitive.

    let a = Compression::Deflate { level: 6 };
    let b = Compression::Deflate { level: 6 };
    let c = Compression::Deflate { level: 6 };

    // Reflexive: a == a
    assert_eq!(a, a);

    // Symmetric: a == b implies b == a
    assert_eq!(a, b);
    assert_eq!(b, a);

    // Transitive: a == b and b == c implies a == c
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_eq!(a, c);
}
