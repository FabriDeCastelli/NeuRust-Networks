//! This module implements containers for a `Dataset``.

/// The trait specifies the methods any `Dataset` must support.
pub trait Dataset {
    type Item;

    /// Pushes a new d-dimensional vector at the end of the `VecDataset``.
    ///
    /// # Panics
    /// Panics if `vec` has an incorrect dimensionality, i.e., `vec.len()` is not d.
    fn push(&mut self, vec: &[Self::Item]);

    /// Pushes new d-dimensional vectors at the end of the dataset.
    ///
    /// # Panics
    /// Panics if `vecs` has an incorrect dimensionality, i.e., `vecs.len()` is not multiple of `d`.
    fn extend(&mut self, vecs: &[Self::Item]);

    /// Returns a random sample of the dataset with `n_vecs` vectors.
    ///
    /// # Panics
    /// Panics if the dataset does not contains enough vectors.
    fn random_sample(&self, n_vecs: usize) -> Self;

    /// Returns an iterator on the vectors of the dataset, one vector at a time.
    fn iter(&self) -> IterVectors<Self::Item> {
        self.iter_batch(1)
    }

    /// Returns an iterator on the vectors in chunks of `batch_size` each,
    /// apart from the last chunk which may be shorter.
    fn iter_batch(&self, batch_size: usize) -> IterVectors<Self::Item>;

    /// Returns the dimensionality of the vectors.
    fn dim(&self) -> usize;

    /// Returns the number of vectors.
    fn len(&self) -> usize;

    /// Returns `true` if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the shape of the dataset.
    fn shape(&self) -> (usize, usize) {
        (self.len(), self.dim())
    }
}

/// A struct to iterate over a dataset. It assumes the dataset can be represented as a slice.
#[derive(Clone)]
pub struct IterVectors<'a, T> {
    v: &'a [T],
    d: usize,
    batch_size: usize,
}

impl<'a, T: 'a> IterVectors<'a, T> {
    /// Creates a new iterator to iterate `batch_size` vectors at a time over the dataset
    /// stored in `slice` with vectors of simensionality `d`.
    pub fn new(slice: &'a [T], d: usize, batch_size: usize) -> Self {
        Self {
            v: slice,
            d,
            batch_size,
        }
    }
}

impl<'a, T> Iterator for IterVectors<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let size = std::cmp::min(self.v.len(), self.batch_size * self.d);
            let (fst, snd) = self.v.split_at(size);
            self.v = snd;
            Some(fst)
        }
    }
}

/// Represents types that can be used as data types for the Numpy `.npy` format.
///
/// This trait is a marker trait that indicates which Rust types are compatible
/// with Numpy data types. Any type that implements this trait can be used to
/// represent data in the `.npy` format. This trait is intended to be lightweight,
/// and does not require any specific implementation details. The purpose is to
/// provide a clear contract for types that are used in this context.
///
/// The trait bounds enforce that types must be `Debug` (for logging and
/// inspection), `Clone` (to ensure data can be duplicated if needed), and
/// `'static` (to ensure no short-lived references).
pub trait NpyDataType: std::fmt::Debug + Clone + 'static {}

/// Macro to implement the `NpyDataType` trait for one or multiple types.
///
/// This macro provides a convenient way to indicate that certain Rust types
/// are compatible with Numpy data types. It reduces boilerplate by allowing
/// multiple types to be passed at once, and then generates the necessary
/// implementations for each type.
macro_rules! impl_npy_data_type {
    ($($t:ty),*) => {
        $(
            impl NpyDataType for $t {}
        )*
    };
}

impl_npy_data_type!(f32, i32);
