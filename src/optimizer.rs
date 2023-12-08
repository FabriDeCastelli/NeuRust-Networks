#[derive(Debug)]
pub enum Optimizer {
    SGD,
    Momentum,
}

impl Optimizer {
    pub fn update(&self) {
        match self {
            Optimizer::SGD => println!("SGD"),
            Optimizer::Momentum => println!("Momentum"),
        }
    }
}
