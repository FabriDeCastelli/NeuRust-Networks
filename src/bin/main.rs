use ML_framework::activation::Activation;
use ML_framework::datasets::vec_dataset::VecDataset;
use ML_framework::loss::Loss;
use ML_framework::metric::Metric;
use ML_framework::models::layers::dense::{DenseBuilder};
use ML_framework::models::sequential::Sequential;
use ML_framework::optimizer::Optimizer;

fn main() {

    // Test 1: [3, 1] NaN
    // Test 2: [10, 5, 3, 1] overfitting + NaN
    let mut model = Sequential::new();
    let dense1 = DenseBuilder::new()
        .input_size(1)
        .output_size(3)
        .activation(Activation::ReLU)
        .build();

    let dense2 = DenseBuilder::new()
        .input_size(3)
        .output_size(1)
        .activation(Activation::Identity)
        .build();

    model.add(dense1);
    model.add(dense2);

    model.compile(
        Optimizer::SGD,
        Loss::MeanSquaredError,
        vec![Metric::MeanSquaredError],
    );

    let x: VecDataset<f32> = VecDataset::<f32>::read_2D_npy("data_for_testing/y=2x/training.npy")
        .expect("Failed to read training data");
    let y: VecDataset<f32> = VecDataset::<f32>::read_2D_npy("data_for_testing/y=2x/labels.npy")
        .expect("Failed to read labels data");

    let (x_train, y_train, x_test, y_test) = VecDataset::train_test_split(x, y, 0.8);

    model.summary();

    // Before training
    // model.print_params();

    model.fit(x_train, y_train, 1000, 20, 0.01, true);

    // After training
    // model.print_params();

    let error = model.evaluate(x_test, y_test);

    println!("Error on the test set: {}", error);
}
