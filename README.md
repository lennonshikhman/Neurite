# Neurite

Neurite is a single-header, C++ machine learning library inspired by Python’s scikit-learn. It provides a wide range of ML algorithms, preprocessing tools, pipeline utilities, and model selection functions—all organized under a clean, consistent API designed for high-performance computing. Neurite leverages Eigen for fast linear algebra operations and is designed to be easily integrated into any C++ project.

## Features

- **Supervised Learning Models:**
  - Linear models (Linear Regression, Logistic Regression, Ridge, Lasso, Elastic Net)
  - Support Vector Machines (SVC, SVR, LinearSVC, LinearSVR, One-Class SVM)
  - Decision Trees and Random Forests
  - Ensemble methods (Gradient Boosting, AdaBoost, Bagging)

- **Unsupervised Learning:**
  - Clustering (K-Means, DBSCAN, Agglomerative Clustering)
  - Dimensionality Reduction (PCA, Truncated SVD, NMF, t-SNE, Isomap)

- **Preprocessing Utilities:**
  - StandardScaler and MinMaxScaler
  - OneHotEncoder and LabelEncoder
  - SimpleImputer for missing values
  - PolynomialFeatures for feature expansion

- **Pipeline and Model Selection:**
  - Pipeline class for chaining transformations and estimators
  - Utility functions for train/test splitting, cross-validation, and grid search

## Installation

Neurite is a header-only library. To use it in your project, simply clone or download the repository and include the header file:

```cpp
#include "neurite.hpp"
```

Neurite depends on [Eigen](https://eigen.tuxfamily.org). Make sure Eigen is installed on your system. You can install it via your package manager or add it as a submodule.

## Build Instructions

Neurite uses CMake for building tests and examples. A sample `CMakeLists.txt` is provided.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Neurite.git
   cd Neurite
   ```

2. **Configure and build:**

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. **Run Tests:**

   If you have added tests in `tests/test_neurite.cpp`, you can run the test executable:

   ```bash
   ./test_neurite
   ```

## Usage Example

Below is a simple example of using Neurite to train a linear regression model:

```cpp
#include "neurite.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // Create some sample data
    Eigen::MatrixXd X(5, 2);
    X << 1, 2,
         2, 3,
         3, 4,
         4, 5,
         5, 6;
    Eigen::VectorXd y(5);
    y << 3, 5, 7, 9, 11;  // y = 1 * x1 + 1 * x2 (for example)

    // Use a linear regression model from Neurite
    sklearn::linear_model::LinearRegression lr;
    lr.fit(X, y);
    Eigen::VectorXd predictions = lr.predict(X);

    std::cout << "Predictions:\n" << predictions << std::endl;
    return 0;
}
```

Compile this example with your C++ compiler (make sure to include the Eigen headers):

```bash
g++ -std=c++17 -I/path/to/Eigen neurite_example.cpp -o neurite_example
./neurite_example
```

## Documentation

Neurite is documented inline using Doxygen-style comments. To generate HTML documentation, configure Doxygen with the provided `docs/Doxyfile` and run:

```bash
doxygen docs/Doxyfile
```

The generated documentation will be in the `docs/html` folder.

## Contributing

Contributions are welcome! If you’d like to contribute to Neurite, please fork the repository and open a pull request. Ensure that new features are well documented and that tests are provided.

## License

Neurite is released under the [MIT License](LICENSE).

## Contact

For any questions or issues, please open an issue on GitHub.
