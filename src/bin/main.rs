use ML_framework::activation::Activation;
use ML_framework::datasets::vec_dataset::VecDataset;
use ML_framework::init_method::InitMethod;
use ML_framework::loss::Loss;
use ML_framework::metric::Metric;
use ML_framework::models::layers::dense::Dense;
use ML_framework::models::sequential::Sequential;
use ML_framework::optimizer::Optimizer;

fn main() {
    let mut model = Sequential::new();

    model.add(Dense::new(1, 3, Activation::ReLU, InitMethod::Random));
    model.add(Dense::new(3, 1, Activation::ReLU, InitMethod::Random));
    model.compile(
        Optimizer::SGD,
        Loss::MeanEuclideanError,
        vec![Metric::MeanSquaredError, Metric::MeanSquaredError],
    );

    let x: VecDataset<f32> = VecDataset::<f32>::read_2D_npy("data_for_testing/y=2x/training.npy")
        .expect("Failed to read training data");
    let y: VecDataset<f32> = VecDataset::<f32>::read_2D_npy("data_for_testing/y=2x/labels.npy")
        .expect("Failed to read labels data");

    let (x_train, y_train, x_test, y_test) =
        VecDataset::train_test_split(x, y, 0.8);

    model.summary();

    model.fit(x_train, y_train, 1000, 20, 0.01, true);
    println!("MSE: {:?}", model.evaluate(x_test, y_test));
}
