use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read, Result as IoResult};
use std::ops::{Add, Mul, Sub};
use std::path::Path;

use crate::dataset::{Dataset, IterVectors, NpyDataType};
use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, ReadableElement, WritableElement, WriteNpyExt};
use rand::prelude::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct VecDataset<T: Copy + Mul + Add + Sub> {
    n_vecs: usize,
    d: usize,
    data: Vec<T>,
}

impl<T: Copy + Mul + Add + Sub> VecDataset<T> {
    /// Creates an empty `VecDataset` object for `d`-dimensional vectors.
    ///
    /// # Arguments
    /// - `d` (`usize`): The dimensionality of the vectors that this dataset
    ///   will contain.
    ///
    /// # Returns
    /// - `VecDataset`: A newly created empty dataset with the specified
    ///   dimensionality.
    ///
    /// # Examples
    /// ```
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let data: VecDataset<f32> = VecDataset::with_dim(100);
    ///
    /// assert_eq!(data.len(), 0);
    /// assert_eq!(data.dim(), 100);
    /// ```
    pub fn with_dim(d: usize) -> Self {
        Self {
            n_vecs: 0,
            d,
            data: Vec::new(),
        }
    }

    /// Creates an empty `VecDataset` object for `d`-dimensional vectors
    /// with enough space to place `n_vecs` vectors.
    ///
    /// # Arguments
    /// - `n_vecs` (`usize`): The number of vectors for which to pre-allocate
    ///   space.
    /// - `d` (`usize`): The dimensionality of the vectors that this dataset
    ///   will contain.
    ///
    /// # Returns
    /// - `VecDataset`: A newly created empty dataset with the specified
    ///   capacity.
    ///
    /// # Examples
    /// ```
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let dataset = VecDataset::<f32>::with_capacity(10, 5);
    /// // Cannot directly check the capacity of the private `data` field
    /// ```
    pub fn with_capacity(n_vecs: usize, d: usize) -> Self {
        Self {
            n_vecs: 0,
            d,
            data: Vec::with_capacity(n_vecs * d),
        }
    }

    /// Reads a `VecDataset` from a file in fvecs format.
    /// The format is explained [here](http://corpus-texmex.irisa.fr).
    ///
    /// # Arguments
    /// - `fname` (`&str`): The path to the file in fvecs format.
    ///
    /// # Returns
    /// - `IoResult<VecDataset<f32>>`: The dataset loaded from the file or an
    ///   error.
    ///
    /// # Examples
    /// ```no_run
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let dataset = VecDataset::<f32>::read_fvecs_file("data.fvecs")
    ///     .expect("Failed to read fvecs file");
    /// ```
    pub fn read_fvecs_file(fname: &str) -> IoResult<VecDataset<f32>> {
        let path = Path::new(fname);
        let f = File::open(path)?;
        let f_size = f.metadata().unwrap().len() as usize;

        let mut br = BufReader::new(f);

        let mut buffer_d = [0u8; std::mem::size_of::<u32>()];
        let mut buffer = [0u8; std::mem::size_of::<f32>()];

        br.read_exact(&mut buffer_d)?;
        let d = u32::from_le_bytes(buffer_d) as usize;

        let n_rows = (f_size) / (d * std::mem::size_of::<f32>() + 4); // f_size = (4+d*value_size)*n ==> n
        let mut data = VecDataset::<f32>::with_capacity(n_rows, d);

        let mut curr_row = vec![0.0; d];
        for row in 0..n_rows {
            if row != 0 {
                br.read_exact(&mut buffer_d)?;
            }
            for item in curr_row.iter_mut().take(d) {
                br.read_exact(&mut buffer)?;
                let v = f32::from_le_bytes(buffer);
                *item = v;
            }
            data.push(&curr_row);
        }

        Ok(data)
    }

    /// Constructs a `VecDataset` from a given vector, number of vectors, and
    /// dimensionality.
    ///
    /// # Arguments
    /// - `v` (`Vec<T>`): A vector containing the dataset values. Its length
    ///   should be `n * d`.
    /// - `n` (`usize`): The number of vectors in the dataset.
    /// - `d` (`usize`): The dimensionality of each vector in the dataset.
    ///
    /// # Panics
    /// - This function will panic if the length of `v` is not equal to `n * d`.
    ///
    /// # Returns
    /// - `VecDataset`: A newly constructed `VecDataset` containing the provided data.
    ///
    /// # Examples
    /// ```
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let vec_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let dataset = VecDataset::<f32>::from_vec(vec_data, 2, 3);
    ///
    /// assert_eq!(dataset.len(), 2);
    /// assert_eq!(dataset.dim(), 3);
    /// ```
    pub fn from_vec(v: Vec<T>, n: usize, d: usize) -> Self {
        assert_eq!(n * d, v.len());

        Self {
            n_vecs: n,
            d,
            data: v,
        }
    }

    /// Generates a `VecDataset` with random values.
    ///
    /// This function creates a `VecDataset` with `n` vectors, each of
    /// dimension `d`, and with each value being a random floating-point
    /// number between 0 and 1.
    ///
    /// # Arguments
    /// - `n` (`usize`): The number of vectors in the dataset.
    /// - `d` (`usize`): The dimensionality of each vector in the dataset.
    ///
    /// # Returns
    /// - `VecDataset<f32>`: A newly constructed `VecDataset` with random data.
    ///
    /// # Examples
    /// ```
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let dataset = VecDataset::<f32>::random(10, 5);
    ///
    /// assert_eq!(dataset.len(), 10);
    /// assert_eq!(dataset.dim(), 5);
    /// ```
    pub fn random(n: usize, d: usize) -> VecDataset<f32> {
        let mut rng = rand::thread_rng();
        let v: Vec<f32> = (0..n * d).map(|_| rng.gen()).collect();

        VecDataset::<f32>::from_vec(v, n, d)
    }

    /// Reads a 2D numpy array from a given file and returns it as a
    /// `VecDataset`.
    ///
    /// The function opens the specified file, reads the numpy array, and
    /// constructs a `VecDataset` from the flattened 2D numpy array.
    ///
    /// # Arguments
    /// - `fname` (`&str`): The file name or path of the numpy file to be read.
    ///
    /// # Returns
    /// - `Result<VecDataset<U>, Box<dyn Error>>`: A result containing
    ///   `VecDataset` or an error.
    ///
    /// # Examples
    /// ```no_run
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let dataset: VecDataset<f32> = VecDataset::<f32>::read_2D_npy("data.npy")
    ///     .expect("Failed to read numpy file");
    /// ```
    #[allow(non_snake_case)]
    pub fn read_2D_npy<
        U: NpyDataType + Copy + ReadableElement + WritableElement + Mul + Add + Sub,
    >(
        fname: &str,
    ) -> Result<VecDataset<U>, Box<dyn Error>> {
        let reader = File::open(fname)?;
        let arr: Array2<U> = Array2::read_npy(reader).map_err(|e| Box::new(e) as Box<dyn Error>)?;

        let mut data = VecDataset::<U>::with_capacity(arr.nrows(), arr.ncols());
        let flat_data: Vec<U> = arr.iter().cloned().collect();
        data.extend(&flat_data);

        Ok(data)
    }

    /// Generates a `VecDataset` of specified size with a given constant value.
    ///
    /// The function creates a 2D ndarray with the specified number of rows
    /// and columns (same as rows), filled with the given value. It then
    /// constructs a `VecDataset` from the flattened ndarray.
    ///
    /// # Arguments
    /// - `rows` (`usize`): The number of rows (and columns) for the 2D ndarray.
    /// - `value` (`U`): The constant value to fill the 2D ndarray.
    ///
    /// # Returns
    /// - `Result<VecDataset<U>, Box<dyn Error>>`: A result containing `VecDataset` or an error.
    ///
    /// # Examples
    /// ```rust
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let dataset = VecDataset::<f32>::get_dataset(10, 0.0f32)
    ///     .expect("Failed to create dataset");
    /// ```
    pub fn get_dataset<U>(rows: usize, value: U) -> Result<VecDataset<U>, Box<dyn Error>>
    where
        U: Copy + 'static + Mul + Add + Sub,
    {
        let data = Array2::from_elem((rows, rows), value);
        let flattened_data: Vec<U> = data.iter().cloned().collect();

        Ok(VecDataset {
            n_vecs: rows,
            d: rows,
            data: flattened_data,
        })
    }

    /// Shrinks the size of allocated memory to fit the dataset size.
    ///
    /// # Examples
    /// ```
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let mut dataset = VecDataset::<f32>::with_capacity(10, 5);
    /// dataset.shrink_to_fit();
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Returns the dimensionality of the vectors in the dataset.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Returns the number of vectors in the dataset.
    pub fn len(&self) -> usize {
        self.n_vecs
    }

    /// Returns a reference to the data vector.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }
}

impl<T> VecDataset<T>
where
    T: Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + WritableElement + Default,
{
    /// Saves the given VecDataset to a .npy file.
    ///
    /// This function takes a reference to a `VecDataset<T>` and a file name as arguments,
    /// and writes the dataset to a file in the .npy format. The .npy format is a simple
    /// format for saving numerical arrays to disk.
    ///
    /// The function is defined outside the `VecDataset` implementation to keep the `VecDataset`
    /// definition clean and to adhere to the user's request of not modifying the existing `VecDataset` code.
    /// This also allows for better separation of concerns, where file IO operations are handled
    /// outside the core `VecDataset` implementation.
    ///
    /// # Type Parameters
    ///
    /// - `T: Copy + WritableElement` - The type of elements in the VecDataset.
    ///    - `Copy` ensures that `T` can be copied, which is necessary for creating the ndarray.
    ///    - `WritableElement` is a trait from the `ndarray_npy` crate that provides the necessary
    ///      functionality to write `T` to a .npy file.
    ///
    /// # Arguments
    ///
    /// - `dataset: &VecDataset<T>` - A reference to the VecDataset to be saved.
    /// - `fname: &str` - The name of the file to save the dataset to. This should include the
    ///    .npy extension.
    ///
    /// # Returns
    ///
    /// - `IoResult<()>` - An `Ok(())` value is returned if the dataset is successfully saved to disk.
    ///    If an error occurs during this process, an `Err` variant containing a `std::io::Error` is returned.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The `VecDataset` data cannot be converted to an `ndarray::Array2`.
    /// - A file cannot be created with the specified name.
    /// - There is an error writing to the .npy file.
    ///
    /// # Examples
    ///
    /// ```
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let dataset: VecDataset<f32> = VecDataset::with_dim(100);
    /// let filename = "dataset.npy";
    ///
    /// if let Err(e) = dataset.to_npy(filename) {
    ///     eprintln!("Error saving dataset: {}", e);
    /// }
    /// ```
    pub fn to_npy(&self, fname: &str) -> IoResult<()> {
        let arr_data = Array2::from_shape_vec((self.n_vecs, self.d), self.data.clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let path = Path::new(fname);
        let writer = File::create(path)?;
        arr_data
            .write_npy(writer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Returns a new VecDataset<T> with the dot product of the input VecDataset<T>.
    ///
    /// # Arguments
    ///
    /// * `x`: the input VecDataset<T>
    ///
    /// returns: VecDataset<T>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn dot(&self, x: &VecDataset<T>) -> VecDataset<T> {
        // println!("dot, {} {}", self.dim(), x.len());
        assert_eq!(self.dim(), x.len(), "Wrong dimensions when multiplying");

        let k = x.dim();

        let mut result = VecDataset::<T>::with_capacity(self.len(), x.dim());

        self.iter().for_each(|row| {
            let mut row_array = Vec::with_capacity(k);
            for j in 0..k {
                let sum = x.as_ref()[j..]
                    .iter()
                    .step_by(k)
                    .zip(row)
                    .fold(T::default(), |acc, (&x, &y)| acc + x * y);

                row_array.push(sum);
            }

            result.push(&row_array);
        });

        result
    }

    pub fn plus(&self, x: &VecDataset<T>) -> VecDataset<T> {
        assert_eq!(self.dim(), x.dim(), "Wrong number of columns when adding");
        assert_eq!(1, x.len(), "Addend not a vector");

        let mut result = Vec::with_capacity(self.len() * self.dim());
        for i in 0..self.len() {
            for j in 0..self.dim() {
                result.push(self.data[i * self.dim() + j] + x.as_ref()[j]);
            }
        }

        VecDataset::from_vec(result, self.len(), self.dim())
    }
    pub fn minus(&self, x: &VecDataset<T>) -> VecDataset<T> {
        assert_eq!(self.len(), x.len(), "Wrong number of rows when subtracting");
        assert_eq!(self.dim(), x.dim(), "Wrong dimensions when subtracting");

        let mut result = Vec::with_capacity(self.len() * self.dim());
        self.iter().zip(x.iter()).for_each(|(v1, v2)| {
            v1.iter().zip(v2.iter()).for_each(|(&a, &b)| {
                result.push(a - b);
            });
        });
        VecDataset::from_vec(result, self.len(), self.dim())
    }

    pub fn times(&self, x: T) -> VecDataset<T> {
        let mut result = Vec::with_capacity(self.len() * self.dim());
        self.iter().for_each(|v| {
            v.iter().for_each(|&a| {
                result.push(a * x);
            });
        });
        VecDataset::from_vec(result, self.len(), self.dim())
    }

    pub fn element_wise_mul(&self, x: &VecDataset<T>) -> VecDataset<T> {
        assert_eq!(self.len(), x.len(), "Wrong number of rows when multiplying");
        assert_eq!(self.dim(), x.dim(), "Wrong dimensions when multiplying");

        let mut result = Vec::with_capacity(self.len() * self.dim());
        self.iter().zip(x.iter()).for_each(|(v1, v2)| {
            v1.iter().zip(v2.iter()).for_each(|(&a, &b)| {
                result.push(a * b);
            });
        });
        VecDataset::from_vec(result, self.len(), self.dim())
    }

    pub fn transpose(&mut self) -> VecDataset<T> {
        if self.len() == 1 || self.dim() == 1 {
            let clone = self.data.clone();
            return VecDataset::from_vec(clone, self.dim(), self.len());
        }

        let mut transposed = Vec::with_capacity(self.len() * self.dim());
        for i in 0..self.len() {
            for j in 0..self.dim() {
                transposed.push(self.data[j * self.dim() + i]);
            }
        }
        VecDataset::from_vec(transposed, self.dim(), self.len())
    }

    pub fn sum_axis(&self, axis: usize) -> VecDataset<T> {
        assert!(axis < 2, "Axis must be 0 or 1");
        let mut result = Vec::with_capacity(self.len() * self.dim());
        match axis {
            0 => {
                for i in 0..self.dim() {
                    let mut sum = T::default();
                    for j in 0..self.len() {
                        sum = sum + self.data[j * self.dim() + i];
                    }
                    result.push(sum);
                }
                VecDataset::from_vec(result, 1, self.dim())
            }
            1 => {
                for i in 0..self.len() {
                    let mut sum = T::default();
                    for j in 0..self.dim() {
                        sum = sum + self.data[i * self.dim() + j];
                    }
                    result.push(sum);
                }
                VecDataset::from_vec(result, self.len(), 1)
            }
            _ => unreachable!(),
        }
    }

    pub fn train_test_split(
        x: VecDataset<T>,
        y: VecDataset<T>,
        test_size: f32,
    ) -> (VecDataset<T>, VecDataset<T>, VecDataset<T>, VecDataset<T>) {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.shuffle(&mut rng);
        let test_size = (x.len() as f32 * test_size) as usize;
        let train_size = x.len() - test_size;
        let mut x_train: VecDataset<T> = VecDataset::with_capacity(train_size, x.dim());
        let mut y_train: VecDataset<T> = VecDataset::with_capacity(train_size, y.dim());
        let mut x_test: VecDataset<T> = VecDataset::with_capacity(test_size, x.dim());
        let mut y_test: VecDataset<T> = VecDataset::with_capacity(test_size, y.dim());
        for i in 0..train_size {
            x_train.push(&x.data[indices[i] * x.dim()..(indices[i] + 1) * x.dim()]);
            y_train.push(&y.data[indices[i] * y.dim()..(indices[i] + 1) * y.dim()]);
        }
        for i in train_size..x.len() {
            x_test.push(&x.data[indices[i] * x.dim()..(indices[i] + 1) * x.dim()]);
            y_test.push(&y.data[indices[i] * y.dim()..(indices[i] + 1) * y.dim()]);
        }
        (x_train, y_train, x_test, y_test)
    }
}
/// AsRef to &[T]
impl<T: Copy + Mul + Add + Sub> AsRef<[T]> for VecDataset<T> {
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<T: Copy + Mul + Add + Sub> Dataset for VecDataset<T> {
    type Item = T;

    /// Pushes a new d-dimensional vector at the end of the `VecDataset``.
    ///
    /// # Panics
    /// Panics if `vec` has an incorrect dimensionality, i.e., is not self.dim().
    /// As it uses a vector, panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    /// ```
    ///
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let mut data = VecDataset::with_dim(128);
    /// let mut vec = vec!(0.0_f32; 128);
    ///
    /// for _ in 0..10 {
    ///     data.push(&vec);
    /// }
    ///
    /// assert_eq!(data.shape(), (10, 128))
    /// ```
    fn push(&mut self, vec: &[Self::Item]) {
        assert_eq!(vec.len(), self.d);

        self.data.extend(vec);

        self.n_vecs += 1;
    }

    /// Pushes new d-dimensional vectors at the end of the `VecDataset`.
    ///
    /// # Panics
    /// Panics if `vecs` has an incorrect dimensionality, i.e., its lenght is not a multiple of
    /// self.dim().
    /// As it uses a vector, panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    /// ```
    /// use ML_framework::datasets::vec_dataset::VecDataset;
    ///
    /// let mut data = VecDataset::with_dim(128);
    /// let mut vec = vec!(0.0_f32; 128*10);
    ///
    /// data.extend(&vec);
    ///
    /// assert_eq!(data.shape(), (10, 128))
    /// ```
    fn extend(&mut self, vecs: &[Self::Item]) {
        assert_eq!(
            vecs.len() % self.d,
            0,
            "The len of vecs must be multiple of the vector dimensionality."
        );

        self.data.extend(vecs);

        self.n_vecs += vecs.len() / self.d;
    }

    /// Returns a `VecDataset` containing a random samples of the dataset with
    /// `n_vec` vectors.
    fn random_sample(&self, n_vecs: usize) -> Self {
        use rand::seq::index::sample;

        let mut rng = rand::thread_rng();
        let sampled_id = sample(&mut rng, self.len(), n_vecs);
        let mut sample = Self::with_capacity(n_vecs, self.d);
        for id in sampled_id {
            sample.push(&self.data[id * self.d..(id + 1) * self.d]);
        }

        sample
    }

    /// Returns an iterator on the vectors in chunks of `batch_size` each,
    /// apart from the last one which may be shorter.
    fn iter_batch(&self, batch_size: usize) -> IterVectors<Self::Item> {
        IterVectors::new(&self.data, self.d, batch_size)
    }

    /// Returns the dimension of the `VecDataset`.
    fn dim(&self) -> usize {
        self.d
    }

    /// Returns the number of vector in the `VecDataset``.
    fn len(&self) -> usize {
        self.n_vecs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_to_npy_and_read_2d_numpy() {
        let nrows = 4;
        let value: f32 = 3.14;
        let fname = "temp_dataset.npy";

        // Create a dataset
        let dataset = VecDataset::<f32>::get_dataset(nrows, value).unwrap();
        assert_eq!(dataset.n_vecs, nrows);
        assert_eq!(dataset.d, nrows);
        assert!(dataset.data.iter().all(|&val| val == value));

        // Convert the dataset to a .npy file
        dataset.to_npy(fname).unwrap();

        // Read the .npy file back into a VecDataset
        let read_dataset: VecDataset<f32> = VecDataset::<f32>::read_2D_npy(fname).unwrap();

        assert_eq!(dataset.n_vecs, read_dataset.n_vecs);
        assert_eq!(dataset.d, read_dataset.d);
        assert_eq!(dataset.data, read_dataset.data);

        std::fs::remove_file(fname).unwrap();
    }

    #[test]
    fn test_dot() {
        let a = vec![0, 1, 2, 3, 4, 5];
        let b = vec![5, 1];
        let dataset1 = VecDataset::<i32>::from_vec(a, 3, 2);
        let dataset2 = VecDataset::<i32>::from_vec(b, 2, 1);
        let result = dataset1.dot(&dataset2);
        let expected = VecDataset::from_vec(vec![1, 13, 25], 3, 1);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_plus() {
        let a = vec![0, 1, 2, 3, 4, 5];
        let b = vec![10, 10];
        let dataset1 = VecDataset::<i32>::from_vec(a, 3, 2);
        let dataset2 = VecDataset::<i32>::from_vec(b, 1, 2);
        let result = dataset1.plus(&dataset2);
        let expected = VecDataset::from_vec(vec![10, 11, 12, 13, 14, 15], 3, 2);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_transpose() {
        let a = vec![1, 0, 5, 0, 1, 0, -5, 0, 1];
        let mut dataset1 = VecDataset::<i32>::from_vec(a, 3, 3);
        let dataset2 = dataset1.transpose();
        let expected = VecDataset::from_vec(vec![1, 0, -5, 0, 1, 0, 5, 0, 1], 3, 3);
        assert_eq!(dataset1.len(), dataset2.dim());
        assert_eq!(dataset1.dim(), dataset2.len());
        assert_eq!(dataset2.data, expected.data);
        let x = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut dataset3 = VecDataset::<i32>::from_vec(x, 1, 9);
        let dataset4 = dataset3.transpose();
        assert_eq!(dataset3.data, dataset4.data);
        assert_eq!(dataset3.len(), dataset4.dim());
        assert_eq!(dataset3.dim(), dataset4.len());
    }
}
