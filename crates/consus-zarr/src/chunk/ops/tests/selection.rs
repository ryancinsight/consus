//! Selection construction and step iteration tests.

use crate::chunk::selection::{Selection, SelectionStep};

#[test]
fn selection_full() {
    let sel = Selection::full(3);
    assert!(sel.is_full());
    assert_eq!(sel.num_elements(), 0);
}

#[test]
fn selection_is_full() {
    let sel = Selection::full(2);
    assert!(sel.is_full());

    let sel = Selection::from_steps(vec![
        SelectionStep {
            start: 0,
            count: 10,
            stride: 1,
        },
        SelectionStep {
            start: 0,
            count: 10,
            stride: 1,
        },
    ]);
    assert!(!sel.is_full());
}

#[test]
fn selection_step_contiguous() {
    let step = SelectionStep {
        start: 0,
        count: 10,
        stride: 1,
    };
    assert!(step.contiguous());

    let step = SelectionStep {
        start: 0,
        count: 10,
        stride: 2,
    };
    assert!(!step.contiguous());
}

#[test]
fn selection_step_end() {
    let step = SelectionStep {
        start: 5,
        count: 3,
        stride: 2,
    };
    assert_eq!(step.end(), 5 + 2 * 2 + 1);
}

#[test]
fn selection_step_indices() {
    let step = SelectionStep {
        start: 0,
        count: 3,
        stride: 2,
    };
    let indices: Vec<u64> = step.indices().collect();
    assert_eq!(indices, vec![0, 2, 4]);
}

#[test]
fn selection_num_elements() {
    let sel = Selection::from_steps(vec![
        SelectionStep {
            start: 0,
            count: 10,
            stride: 1,
        },
        SelectionStep {
            start: 0,
            count: 5,
            stride: 1,
        },
    ]);
    assert_eq!(sel.num_elements(), 50);
}
