#include "neurite.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // --- Test 1: Linear Regression ---
    std::cout << "Testing Linear Regression..." << std::endl;
    // Create a simple dataset: y = 1*x1 + 1*x2
    Eigen::MatrixXd X_lr(5, 2);
    X_lr << 1, 2,
            2, 3,
            3, 4,
            4, 5,
            5, 6;
    Eigen::VectorXd y_lr(5);
    y_lr << 3, 5, 7, 9, 11;

    neurite::linear_model::LinearRegression lr;
    lr.fit(X_lr, y_lr);
    Eigen::VectorXd preds_lr = lr.predict(X_lr);
    std::cout << "LinearRegression Predictions:\n" << preds_lr << "\n\n";

    // --- Test 2: Standard Scaler ---
    std::cout << "Testing StandardScaler..." << std::endl;
    neurite::preprocessing::StandardScaler scaler;
    scaler.fit(X_lr);
    Eigen::MatrixXd X_scaled = scaler.transform(X_lr);
    std::cout << "Original X:\n" << X_lr << "\n";
    std::cout << "Scaled X:\n" << X_scaled << "\n\n";

    // --- Test 3: K-Means Clustering ---
    std::cout << "Testing KMeans Clustering..." << std::endl;
    // Create a simple clustering dataset with two clear clusters
    Eigen::MatrixXd X_cluster(6, 2);
    X_cluster << 1, 2,
                 1, 1,
                 2, 2,
                 10, 10,
                 10, 11,
                 11, 10;
    neurite::cluster::KMeans kmeans(2, 100);
    kmeans.fit(X_cluster);
    Eigen::VectorXi cluster_labels = kmeans.predict(X_cluster);
    std::cout << "KMeans Cluster Labels:\n" << cluster_labels << "\n\n";

    // --- Test 4: Pipeline Example ---
    std::cout << "Testing Pipeline..." << std::endl;
    // For demonstration, chain StandardScaler and LinearRegression in a pipeline
    neurite::pipeline::Pipeline pipe;
    pipe.add_transformer("scaler", neurite::preprocessing::StandardScaler());
    pipe.add_estimator("lr", neurite::linear_model::LinearRegression());
    pipe.fit(X_lr, y_lr);
    Eigen::VectorXd pipe_preds = pipe.predict(X_lr);
    std::cout << "Pipeline Predictions:\n" << pipe_preds << "\n";

    return 0;
}
