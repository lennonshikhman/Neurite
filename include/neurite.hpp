#ifndef NEURITE_HPP
#define NEURITE_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <iterator>
#include <iostream>

// Neurite: A single-header, scikit-learn–like C++ machine learning library

namespace neurite {

//////////////////////////////
// Linear Models
//////////////////////////////
namespace linear_model {

  /** \brief Ordinary Least Squares Linear Regression.
   *  Fits model y = Xw + b minimizing the squared error.
   */
  class LinearRegression {
  public:
    bool fit_intercept;
    Eigen::VectorXd coef_;
    double intercept_;
    
    LinearRegression(bool fit_intercept = true)
      : fit_intercept(fit_intercept), intercept_(0.0) {}
    
    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      if (fit_intercept) {
        Eigen::MatrixXd X1(X.rows(), X.cols() + 1);
        X1 << X, Eigen::VectorXd::Ones(X.rows());
        Eigen::VectorXd w = X1.colPivHouseholderQr().solve(y);
        coef_ = w.head(w.size() - 1);
        intercept_ = w[w.size() - 1];
      } else {
        coef_ = X.colPivHouseholderQr().solve(y);
        intercept_ = 0.0;
      }
    }
    
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = X * coef_;
      if (fit_intercept)
        preds.array() += intercept_;
      return preds;
    }
  };

  /** \brief Ridge Regression (L2-regularized linear regression).
   */
  class Ridge {
  public:
    double alpha;
    bool fit_intercept;
    Eigen::VectorXd coef_;
    double intercept_;
    
    Ridge(double alpha = 1.0, bool fit_intercept = true)
      : alpha(alpha), fit_intercept(fit_intercept), intercept_(0.0) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      int n_features = X.cols();
      Eigen::MatrixXd XtX = X.transpose() * X;
      XtX.diagonal().array() += alpha;
      Eigen::VectorXd Xty = X.transpose() * y;
      Eigen::VectorXd w = XtX.ldlt().solve(Xty);
      coef_ = w;
      intercept_ = 0.0;
      if (fit_intercept) {
        intercept_ = y.mean() - (X.rowwise().mean() * coef_).value();
      }
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = X * coef_;
      if (fit_intercept)
        preds.array() += intercept_;
      return preds;
    }
  };

  /** \brief Lasso Regression (L1-regularized linear regression).
   *  Uses coordinate descent (pseudocode).
   */
  class Lasso {
  public:
    double alpha;
    bool fit_intercept;
    Eigen::VectorXd coef_;
    double intercept_;
    int max_iter;
    
    Lasso(double alpha = 1.0, bool fit_intercept = true, int max_iter = 1000)
      : alpha(alpha), fit_intercept(fit_intercept), intercept_(0.0), max_iter(max_iter) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      // Pseudocode: initialize coefficients to zero and update coordinate-wise
      coef_ = Eigen::VectorXd::Zero(X.cols());
      intercept_ = fit_intercept ? y.mean() : 0.0;
      // Coordinate descent updates would be implemented here.
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = X * coef_;
      if (fit_intercept)
        preds.array() += intercept_;
      return preds;
    }
  };

  /** \brief ElasticNet Regression (combined L1 and L2).
   */
  class ElasticNet {
  public:
    double alpha;
    double l1_ratio;
    bool fit_intercept;
    Eigen::VectorXd coef_;
    double intercept_;
    int max_iter;
    
    ElasticNet(double alpha = 1.0, double l1_ratio = 0.5, bool fit_intercept = true, int max_iter = 1000)
      : alpha(alpha), l1_ratio(l1_ratio), fit_intercept(fit_intercept), intercept_(0.0), max_iter(max_iter) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      coef_ = Eigen::VectorXd::Zero(X.cols());
      intercept_ = fit_intercept ? y.mean() : 0.0;
      // Pseudocode for combined coordinate descent update.
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = X * coef_;
      if (fit_intercept)
        preds.array() += intercept_;
      return preds;
    }
  };

  /** \brief Logistic Regression for classification.
   */
  class LogisticRegression {
  public:
    double C;
    std::string penalty;
    bool fit_intercept;
    int max_iter;
    Eigen::MatrixXd coef_;
    Eigen::VectorXd intercept_;
    std::vector<int> classes_;
    
    LogisticRegression(double C = 1.0, std::string penalty = "l2", bool fit_intercept = true, int max_iter = 100)
      : C(C), penalty(penalty), fit_intercept(fit_intercept), max_iter(max_iter) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      classes_.clear();
      for (int i = 0; i < y.size(); ++i) {
        int label = static_cast<int>(std::round(y[i]));
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end())
          classes_.push_back(label);
      }
      std::sort(classes_.begin(), classes_.end());
      int n_classes = classes_.size();
      int n_features = X.cols();
      coef_ = Eigen::MatrixXd::Zero(n_classes, n_features);
      intercept_ = Eigen::VectorXd::Zero(n_classes);
      // Pseudocode: implement solver (e.g., gradient descent) to optimize log-likelihood.
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::MatrixXd scores = decision_function(X);
      Eigen::VectorXi labels(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        Eigen::VectorXd row = scores.row(i);
        Eigen::Index maxIndex;
        row.maxCoeff(&maxIndex);
        labels[i] = classes_[maxIndex];
      }
      return labels;
    }

    Eigen::MatrixXd predict_proba(const Eigen::MatrixXd &X) const {
      Eigen::MatrixXd scores = decision_function(X);
      Eigen::MatrixXd probs = scores;
      for (int i = 0; i < probs.rows(); ++i) {
        double maxScore = scores.row(i).maxCoeff();
        Eigen::VectorXd expScores = (scores.row(i).array() - maxScore).exp();
        probs.row(i) = expScores / expScores.sum();
      }
      return probs;
    }

  private:
    Eigen::MatrixXd decision_function(const Eigen::MatrixXd &X) const {
      Eigen::MatrixXd scores = (coef_ * X.transpose()).transpose();
      if (fit_intercept) {
        for (int i = 0; i < scores.rows(); ++i)
          scores.row(i) += intercept_.transpose();
      }
      return scores;
    }
  };

} // namespace linear_model

//////////////////////////////
// Support Vector Machines (SVM)
//////////////////////////////
namespace svm {

  /** \brief C-Support Vector Classification.
   */
  class SVC {
  public:
    double C;
    std::string kernel;
    double gamma;
    int degree;
    double coef0;
    Eigen::MatrixXd support_vectors_;
    Eigen::VectorXd dual_coeff_;
    double rho_;
    
    SVC(double C = 1.0, std::string kernel = "rbf", double gamma = 0.1, int degree = 3, double coef0 = 0.0)
      : C(C), kernel(kernel), gamma(gamma), degree(degree), coef0(coef0), rho_(0.0) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      // Placeholder: call to external SVM solver.
      support_vectors_ = X; // Not an actual implementation.
      dual_coeff_ = Eigen::VectorXd::Ones(X.rows());
      rho_ = 0.0;
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi labels(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double decision = 0.0;
        for (int j = 0; j < support_vectors_.rows(); ++j) {
          double kernel_val = support_vectors_.row(j).dot(X.row(i).transpose());
          decision += dual_coeff_[j] * kernel_val;
        }
        decision += rho_;
        labels[i] = (decision >= 0 ? 1 : -1);
      }
      return labels;
    }
  };

  /** \brief Support Vector Regression.
   */
  class SVR {
  public:
    double C;
    std::string kernel;
    double gamma;
    int degree;
    double coef0;
    double epsilon;
    Eigen::MatrixXd support_vectors_;
    Eigen::VectorXd dual_coeff_;
    double rho_;
    
    SVR(double C = 1.0, std::string kernel = "rbf", double gamma = 0.1, int degree = 3, double coef0 = 0.0, double epsilon = 0.1)
      : C(C), kernel(kernel), gamma(gamma), degree(degree), coef0(coef0), epsilon(epsilon), rho_(0.0) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      support_vectors_ = X;
      dual_coeff_ = Eigen::VectorXd::Zero(X.rows());
      rho_ = y.mean();
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double decision = 0.0;
        for (int j = 0; j < support_vectors_.rows(); ++j) {
          double kernel_val = support_vectors_.row(j).dot(X.row(i).transpose());
          decision += dual_coeff_[j] * kernel_val;
        }
        decision += rho_;
        preds[i] = decision;
      }
      return preds;
    }
  };

  /** \brief LinearSVC: linear kernel SVM for classification.
   */
  class LinearSVC {
  public:
    double C;
    bool fit_intercept;
    Eigen::VectorXd coef_;
    double intercept_;
    
    LinearSVC(double C = 1.0, bool fit_intercept = true)
      : C(C), fit_intercept(fit_intercept), intercept_(0.0) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      coef_ = Eigen::VectorXd::Zero(X.cols());
      intercept_ = 0.0;
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi labels(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double score = X.row(i).dot(coef_);
        if (fit_intercept)
          score += intercept_;
        labels[i] = (score >= 0 ? 1 : -1);
      }
      return labels;
    }
  };

  /** \brief LinearSVR: linear kernel SVM for regression.
   */
  class LinearSVR {
  public:
    double C;
    bool fit_intercept;
    double epsilon;
    Eigen::VectorXd coef_;
    double intercept_;
    
    LinearSVR(double C = 1.0, bool fit_intercept = true, double epsilon = 0.1)
      : C(C), fit_intercept(fit_intercept), epsilon(epsilon), intercept_(0.0) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      coef_ = Eigen::VectorXd::Zero(X.cols());
      intercept_ = y.mean();
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = X * coef_;
      if (fit_intercept)
        preds.array() += intercept_;
      return preds;
    }
  };

  /** \brief One-class SVM for outlier/novelty detection.
   */
  class OneClassSVM {
  public:
    double nu;
    std::string kernel;
    double gamma;
    Eigen::MatrixXd support_vectors_;
    Eigen::VectorXd dual_coeff_;
    double rho_;

    OneClassSVM(double nu = 0.5, std::string kernel = "rbf", double gamma = 0.1)
      : nu(nu), kernel(kernel), gamma(gamma), rho_(0.0) {}

    void fit(const Eigen::MatrixXd &X) {
      support_vectors_ = X;
      dual_coeff_ = Eigen::VectorXd::Ones(X.rows());
      rho_ = 0.0;
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi labels = Eigen::VectorXi::Ones(X.rows());
      return labels;
    }
  };

} // namespace svm

//////////////////////////////
// Decision Trees
//////////////////////////////
namespace tree {

  /** \brief Node of a Decision Tree.
   */
  struct TreeNode {
    bool is_leaf;
    int feature_index;
    double threshold;
    double value;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : is_leaf(false), feature_index(-1), threshold(0.0), value(0.0), left(nullptr), right(nullptr) {}
  };

  /** \brief Decision Tree Classifier.
   */
  class DecisionTreeClassifier {
  public:
    std::string criterion; // "gini" or "entropy"
    int max_depth;
    int min_samples_split;
    TreeNode* root;

    DecisionTreeClassifier(std::string criterion = "gini", int max_depth = -1, int min_samples_split = 2)
      : criterion(criterion), max_depth(max_depth), min_samples_split(min_samples_split), root(nullptr) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      root = build_tree(X, y, 0);
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi preds(X.rows());
      for (int i = 0; i < X.rows(); ++i)
        preds[i] = predict_sample(X.row(i), root);
      return preds;
    }

  private:
    TreeNode* build_tree(const Eigen::MatrixXd &X, const Eigen::VectorXi &y, int depth) {
      TreeNode* node = new TreeNode();
      bool all_same = true;
      for (int i = 1; i < y.size(); ++i) {
        if (y[i] != y[0]) { all_same = false; break; }
      }
      if (all_same || (max_depth >= 0 && depth >= max_depth) || y.size() < min_samples_split) {
        node->is_leaf = true;
        node->value = y.size() > 0 ? y[0] : 0;
        return node;
      }
      int best_feat = 0;
      double best_thresh = X.col(0).mean();
      node->feature_index = best_feat;
      node->threshold = best_thresh;
      std::vector<int> left_idx, right_idx;
      for (int i = 0; i < X.rows(); ++i) {
        if (X(i, best_feat) <= best_thresh)
          left_idx.push_back(i);
        else
          right_idx.push_back(i);
      }
      Eigen::MatrixXd X_left(left_idx.size(), X.cols());
      Eigen::VectorXi y_left(left_idx.size());
      for (size_t i = 0; i < left_idx.size(); ++i) {
        X_left.row(i) = X.row(left_idx[i]);
        y_left[i] = y[left_idx[i]];
      }
      Eigen::MatrixXd X_right(right_idx.size(), X.cols());
      Eigen::VectorXi y_right(right_idx.size());
      for (size_t j = 0; j < right_idx.size(); ++j) {
        X_right.row(j) = X.row(right_idx[j]);
        y_right[j] = y[right_idx[j]];
      }
      node->left = build_tree(X_left, y_left, depth + 1);
      node->right = build_tree(X_right, y_right, depth + 1);
      return node;
    }

    int predict_sample(const Eigen::RowVectorXd &x, TreeNode* node) const {
      if (node->is_leaf)
        return static_cast<int>(node->value);
      if (x[node->feature_index] <= node->threshold)
        return predict_sample(x, node->left);
      else
        return predict_sample(x, node->right);
    }
  };

  /** \brief Decision Tree Regressor.
   */
  class DecisionTreeRegressor {
  public:
    std::string criterion; // "mse"
    int max_depth;
    int min_samples_split;
    TreeNode* root;

    DecisionTreeRegressor(std::string criterion = "mse", int max_depth = -1, int min_samples_split = 2)
      : criterion(criterion), max_depth(max_depth), min_samples_split(min_samples_split), root(nullptr) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      root = new TreeNode();
      root->is_leaf = true;
      root->value = y.size() > 0 ? y.mean() : 0.0;
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = Eigen::VectorXd::Zero(X.rows());
      for (int i = 0; i < X.rows(); ++i)
        preds[i] = root->value;
      return preds;
    }
  };

} // namespace tree

//////////////////////////////
// Ensemble Methods
//////////////////////////////
namespace ensemble {

  /** \brief Random Forest Classifier.
   */
  class RandomForestClassifier {
  public:
    int n_estimators;
    int max_depth;
    std::vector<tree::DecisionTreeClassifier> trees;
    
    RandomForestClassifier(int n_estimators = 100, int max_depth = -1)
      : n_estimators(n_estimators), max_depth(max_depth) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      trees.clear();
      std::mt19937 gen(0);
      std::uniform_int_distribution<int> dist(0, X.rows() - 1);
      for (int i = 0; i < n_estimators; ++i) {
        Eigen::MatrixXd X_sample(X.rows(), X.cols());
        Eigen::VectorXi y_sample(X.rows());
        for (int j = 0; j < X.rows(); ++j) {
          int idx = dist(gen);
          X_sample.row(j) = X.row(idx);
          y_sample[j] = y[idx];
        }
        tree::DecisionTreeClassifier dt("gini", max_depth);
        dt.fit(X_sample, y_sample);
        trees.push_back(dt);
      }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi preds = Eigen::VectorXi::Zero(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        std::map<int, int> vote_count;
        for (const auto &dt : trees) {
          int pred = dt.predict(X.row(i))[0];
          vote_count[pred] += 1;
        }
        int best_class = 0, best_count = -1;
        for (auto &p : vote_count) {
          if (p.second > best_count) { best_count = p.second; best_class = p.first; }
        }
        preds[i] = best_class;
      }
      return preds;
    }
  };

  /** \brief Random Forest Regressor.
   */
  class RandomForestRegressor {
  public:
    int n_estimators;
    int max_depth;
    std::vector<tree::DecisionTreeRegressor> trees;
    
    RandomForestRegressor(int n_estimators = 100, int max_depth = -1)
      : n_estimators(n_estimators), max_depth(max_depth) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      trees.clear();
      std::mt19937 gen(0);
      std::uniform_int_distribution<int> dist(0, X.rows() - 1);
      for (int i = 0; i < n_estimators; ++i) {
        Eigen::MatrixXd X_sample(X.rows(), X.cols());
        Eigen::VectorXd y_sample(X.rows());
        for (int j = 0; j < X.rows(); ++j) {
          int idx = dist(gen);
          X_sample.row(j) = X.row(idx);
          y_sample[j] = y[idx];
        }
        tree::DecisionTreeRegressor dt("mse", max_depth);
        dt.fit(X_sample, y_sample);
        trees.push_back(dt);
      }
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = Eigen::VectorXd::Zero(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double sum = 0.0;
        for (const auto &dt : trees) {
          sum += dt.predict(X.row(i))[0];
        }
        preds[i] = sum / trees.size();
      }
      return preds;
    }
  };

  /** \brief Gradient Boosting Classifier.
   */
  class GradientBoostingClassifier {
  public:
    int n_estimators;
    double learning_rate;
    int max_depth;
    std::vector<tree::DecisionTreeRegressor> trees;
    Eigen::VectorXd initial_pred;
    
    GradientBoostingClassifier(int n_estimators = 100, double learning_rate = 0.1, int max_depth = 3)
      : n_estimators(n_estimators), learning_rate(learning_rate), max_depth(max_depth) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      int n = X.rows();
      int K = 2;
      double p = ((double)(y.array() == 1).count()) / n;
      double init_val = std::log(p / (1 - p));
      initial_pred = Eigen::VectorXd::Constant(n, init_val);
      Eigen::VectorXd preds = 1 / (1 + (-initial_pred.array()).exp());
      Eigen::VectorXd residual = Eigen::VectorXd::Zero(n);
      trees.clear();
      for (int m = 0; m < n_estimators; ++m) {
        for (int i = 0; i < n; ++i) {
          residual[i] = y[i] - preds[i];
        }
        tree::DecisionTreeRegressor tree_reg("mse", max_depth);
        tree_reg.fit(X, residual);
        trees.push_back(tree_reg);
        Eigen::VectorXd update = tree_reg.predict(X);
        initial_pred.array() += learning_rate * update.array();
        preds = 1 / (1 + (-initial_pred.array()).exp());
      }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd score = Eigen::VectorXd::Zero(X.rows());
      score.array() += initial_pred[0];
      for (const auto &tree_reg : trees)
        score += learning_rate * tree_reg.predict(X);
      Eigen::VectorXi labels(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double prob = 1 / (1 + std::exp(-score[i]));
        labels[i] = (prob >= 0.5 ? 1 : 0);
      }
      return labels;
    }
  };

  /** \brief Gradient Boosting Regressor.
   */
  class GradientBoostingRegressor {
  public:
    int n_estimators;
    double learning_rate;
    int max_depth;
    std::vector<tree::DecisionTreeRegressor> trees;
    double initial_pred;
    
    GradientBoostingRegressor(int n_estimators = 100, double learning_rate = 0.1, int max_depth = 3)
      : n_estimators(n_estimators), learning_rate(learning_rate), max_depth(max_depth), initial_pred(0.0) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      int n = X.rows();
      initial_pred = y.mean();
      Eigen::VectorXd residual = Eigen::VectorXd::Zero(n);
      Eigen::VectorXd pred = Eigen::VectorXd::Constant(n, initial_pred);
      trees.clear();
      for (int m = 0; m < n_estimators; ++m) {
        residual = y - pred;
        tree::DecisionTreeRegressor tree_reg("mse", max_depth);
        tree_reg.fit(X, residual);
        trees.push_back(tree_reg);
        Eigen::VectorXd update = tree_reg.predict(X);
        pred.array() += learning_rate * update.array();
      }
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd pred = Eigen::VectorXd::Constant(X.rows(), initial_pred);
      for (const auto &tree_reg : trees)
        pred += learning_rate * tree_reg.predict(X);
      return pred;
    }
  };

  /** \brief AdaBoost Classifier.
   */
  class AdaBoostClassifier {
  public:
    int n_estimators;
    double learning_rate;
    std::vector<tree::DecisionTreeClassifier> stumps;
    std::vector<double> stump_weights;
    
    AdaBoostClassifier(int n_estimators = 50, double learning_rate = 1.0)
      : n_estimators(n_estimators), learning_rate(learning_rate) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      int n = X.rows();
      Eigen::VectorXd w = Eigen::VectorXd::Ones(n) / n;
      stumps.clear();
      stump_weights.clear();
      for (int m = 0; m < n_estimators; ++m) {
        tree::DecisionTreeClassifier stump("gini", 1);
        stump.fit(X, y);
        Eigen::VectorXi preds = stump.predict(X);
        double err = 0.0;
        for (int i = 0; i < n; ++i) {
          if (preds[i] != y[i])
            err += w[i];
        }
        if (err > 0.5)
          break;
        double alpha = learning_rate * std::log((1 - err) / std::max(err, 1e-10));
        stump_weights.push_back(alpha);
        stumps.push_back(stump);
        for (int i = 0; i < n; ++i) {
          if (preds[i] == y[i])
            w[i] *= std::exp(-alpha);
          else
            w[i] *= std::exp(alpha);
        }
        w /= w.sum();
      }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd agg = Eigen::VectorXd::Zero(X.rows());
      for (size_t m = 0; m < stumps.size(); ++m) {
        Eigen::VectorXi preds = stumps[m].predict(X);
        for (int i = 0; i < X.rows(); ++i) {
          agg[i] += stump_weights[m] * (preds[i] == 1 ? 1 : -1);
        }
      }
      Eigen::VectorXi labels(X.rows());
      for (int i = 0; i < X.rows(); ++i)
        labels[i] = (agg[i] >= 0 ? 1 : -1);
      return labels;
    }
  };

  /** \brief AdaBoost Regressor.
   */
  class AdaBoostRegressor {
  public:
    int n_estimators;
    double learning_rate;
    std::vector<tree::DecisionTreeRegressor> learners;
    std::vector<double> learner_weights;
    
    AdaBoostRegressor(int n_estimators = 50, double learning_rate = 1.0)
      : n_estimators(n_estimators), learning_rate(learning_rate) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      int n = X.rows();
      Eigen::VectorXd pred = Eigen::VectorXd::Zero(n);
      learners.clear();
      learner_weights.clear();
      Eigen::VectorXd w = Eigen::VectorXd::Ones(n) / n;
      for (int m = 0; m < n_estimators; ++m) {
        tree::DecisionTreeRegressor stump("mse", 1);
        stump.fit(X, y);
        Eigen::VectorXd preds = stump.predict(X);
        double err = ((y - preds).cwiseAbs().cwiseProduct(w)).sum();
        if (err >= 0.5)
          break;
        double beta = err / (1 - err);
        double alpha = std::log(1 / beta);
        learners.push_back(stump);
        learner_weights.push_back(alpha);
        for (int i = 0; i < n; ++i) {
          w[i] *= std::exp(alpha * std::fabs(y[i] - preds[i]));
        }
        w /= w.sum();
      }
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = Eigen::VectorXd::Zero(X.rows());
      if (learners.empty())
        return preds;
      double totalWeight = 0.0;
      for (double alpha : learner_weights)
        totalWeight += alpha;
      for (size_t m = 0; m < learners.size(); ++m)
        preds += (learner_weights[m] / totalWeight) * learners[m].predict(X);
      return preds;
    }
  };

  /** \brief Bagging Classifier (template for any base estimator).
   */
  template<typename BaseEstimator = tree::DecisionTreeClassifier>
  class BaggingClassifier {
  public:
    int n_estimators;
    std::vector<BaseEstimator> models;
    BaggingClassifier(int n_estimators = 10) : n_estimators(n_estimators) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      models.clear();
      std::mt19937 gen(0);
      std::uniform_int_distribution<int> dist(0, X.rows() - 1);
      for (int i = 0; i < n_estimators; ++i) {
        Eigen::MatrixXd X_sample(X.rows(), X.cols());
        Eigen::VectorXi y_sample(X.rows());
        for (int j = 0; j < X.rows(); ++j) {
          int idx = dist(gen);
          X_sample.row(j) = X.row(idx);
          y_sample[j] = y[idx];
        }
        BaseEstimator model;
        model.fit(X_sample, y_sample);
        models.push_back(model);
      }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi preds = Eigen::VectorXi::Zero(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        std::map<int, int> vote_count;
        for (const auto &model : models) {
          int pred = model.predict(X.row(i))[0];
          vote_count[pred] += 1;
        }
        int best_class = 0, best_count = -1;
        for (auto &p : vote_count) {
          if (p.second > best_count) { best_count = p.second; best_class = p.first; }
        }
        preds[i] = best_class;
      }
      return preds;
    }
  };

  /** \brief Bagging Regressor (template for any base estimator).
   */
  template<typename BaseEstimator = tree::DecisionTreeRegressor>
  class BaggingRegressor {
  public:
    int n_estimators;
    std::vector<BaseEstimator> models;
    BaggingRegressor(int n_estimators = 10) : n_estimators(n_estimators) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      models.clear();
      std::mt19937 gen(0);
      std::uniform_int_distribution<int> dist(0, X.rows() - 1);
      for (int i = 0; i < n_estimators; ++i) {
        Eigen::MatrixXd X_sample(X.rows(), X.cols());
        Eigen::VectorXd y_sample(X.rows());
        for (int j = 0; j < X.rows(); ++j) {
          int idx = dist(gen);
          X_sample.row(j) = X.row(idx);
          y_sample[j] = y[idx];
        }
        BaseEstimator model;
        model.fit(X_sample, y_sample);
        models.push_back(model);
      }
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds = Eigen::VectorXd::Zero(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double sum = 0.0;
        for (const auto &model : models)
          sum += model.predict(X.row(i))[0];
        preds[i] = sum / models.size();
      }
      return preds;
    }
  };

} // namespace ensemble

//////////////////////////////
// Nearest Neighbors
//////////////////////////////
namespace neighbors {

  /** \brief K-Nearest Neighbors Classifier.
   */
  class KNeighborsClassifier {
  public:
    int n_neighbors;
    std::string weights;
    Eigen::MatrixXd X_train;
    Eigen::VectorXi y_train;
    
    KNeighborsClassifier(int n_neighbors = 5, std::string weights = "uniform")
      : n_neighbors(n_neighbors), weights(weights) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      X_train = X;
      y_train = y;
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi preds(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        Eigen::VectorXd dists(X_train.rows());
        for (int j = 0; j < X_train.rows(); ++j) {
          dists[j] = (X_train.row(j) - X.row(i)).squaredNorm();
        }
        std::vector<int> idx(X_train.rows());
        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin() + n_neighbors, idx.end(), [&](int a, int b) { return dists[a] < dists[b]; });
        idx.resize(n_neighbors);
        std::map<int, double> vote_weight;
        for (int j : idx) {
          double weight = 1.0;
          if (weights == "distance")
            weight = 1.0 / (1e-9 + std::sqrt(dists[j]));
          vote_weight[y_train[j]] += weight;
        }
        int best_label = y_train[idx[0]];
        double best_weight = -1.0;
        for (auto &p : vote_weight) {
          if (p.second > best_weight) { best_weight = p.second; best_label = p.first; }
        }
        preds[i] = best_label;
      }
      return preds;
    }
  };

  /** \brief K-Nearest Neighbors Regressor.
   */
  class KNeighborsRegressor {
  public:
    int n_neighbors;
    std::string weights;
    Eigen::MatrixXd X_train;
    Eigen::VectorXd y_train;
    
    KNeighborsRegressor(int n_neighbors = 5, std::string weights = "uniform")
      : n_neighbors(n_neighbors), weights(weights) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      X_train = X;
      y_train = y;
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXd preds(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        Eigen::VectorXd dists(X_train.rows());
        for (int j = 0; j < X_train.rows(); ++j) {
          dists[j] = (X_train.row(j) - X.row(i)).squaredNorm();
        }
        std::vector<int> idx(X_train.rows());
        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin() + n_neighbors, idx.end(), [&](int a, int b) { return dists[a] < dists[b]; });
        idx.resize(n_neighbors);
        double sum = 0.0, total_weight = 0.0;
        for (int j : idx) {
          double weight = 1.0;
          if (weights == "distance")
            weight = 1.0 / (1e-9 + std::sqrt(dists[j]));
          sum += weight * y_train[j];
          total_weight += weight;
        }
        preds[i] = sum / total_weight;
      }
      return preds;
    }
  };

} // namespace neighbors

//////////////////////////////
// Naive Bayes
//////////////////////////////
namespace naive_bayes {

  /** \brief Gaussian Naive Bayes Classifier.
   */
  class GaussianNB {
  public:
    std::vector<int> classes_;
    std::map<int, double> class_prior_;
    std::map<int, Eigen::VectorXd> mean_;
    std::map<int, Eigen::VectorXd> var_;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      classes_.clear();
      class_prior_.clear();
      mean_.clear();
      var_.clear();
      int n = X.rows(), m = X.cols();
      std::map<int, int> count;
      for (int i = 0; i < n; ++i)
        count[y[i]]++;
      for (auto &p : count)
        classes_.push_back(p.first);
      for (int c : classes_) {
        class_prior_[c] = (double)count[c] / n;
        Eigen::MatrixXd X_c(count[c], m);
        int idx = 0;
        for (int i = 0; i < n; ++i) {
          if (y[i] == c)
            X_c.row(idx++) = X.row(i);
        }
        Eigen::VectorXd mu = X_c.colwise().mean();
        Eigen::VectorXd var_vec = Eigen::VectorXd::Zero(m);
        for (int j = 0; j < X_c.rows(); ++j) {
          Eigen::VectorXd diff = X_c.row(j).transpose() - mu;
          var_vec += diff.array().square().matrix();
        }
        var_vec /= X_c.rows();
        mean_[c] = mu;
        var_[c] = var_vec;
      }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi preds(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double best_log_prob = -1e9;
        int best_class = classes_[0];
        for (int c : classes_) {
          double log_prob = std::log(class_prior_.at(c));
          const Eigen::VectorXd &mu = mean_.at(c);
          const Eigen::VectorXd &var_vec = var_.at(c);
          for (int j = 0; j < X.cols(); ++j) {
            double x = X(i, j);
            double sigma2 = var_vec[j] + 1e-9;
            log_prob += -0.5 * std::log(2 * M_PI * sigma2) - (x - mu[j]) * (x - mu[j]) / (2 * sigma2);
          }
          if (log_prob > best_log_prob) {
            best_log_prob = log_prob;
            best_class = c;
          }
        }
        preds[i] = best_class;
      }
      return preds;
    }
  };

  /** \brief Multinomial Naive Bayes Classifier.
   */
  class MultinomialNB {
  public:
    double alpha;
    std::vector<int> classes_;
    std::map<int, double> class_prior_;
    std::map<int, Eigen::VectorXd> feature_log_prob_;

    MultinomialNB(double alpha = 1.0) : alpha(alpha) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      classes_.clear();
      class_prior_.clear();
      feature_log_prob_.clear();
      int n = X.rows(), m = X.cols();
      std::map<int, int> count;
      for (int i = 0; i < n; ++i)
        count[y[i]]++;
      for (auto &p : count)
        classes_.push_back(p.first);
      for (int c : classes_) {
        class_prior_[c] = (double)count[c] / n;
        Eigen::VectorXd feature_count = Eigen::VectorXd::Zero(m);
        double total_count = 0.0;
        for (int i = 0; i < n; ++i) {
          if (y[i] == c) {
            feature_count += X.row(i).transpose();
            total_count += X.row(i).sum();
          }
        }
        Eigen::VectorXd log_prob(m);
        for (int j = 0; j < m; ++j) {
          double count_wc = feature_count[j];
          log_prob[j] = std::log((count_wc + alpha) / (total_count + alpha * m));
        }
        feature_log_prob_[c] = log_prob;
      }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi preds(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double best_log_prob = -1e9;
        int best_class = classes_[0];
        for (int c : classes_) {
          double log_prob = std::log(class_prior_.at(c));
          const Eigen::VectorXd &log_prob_c = feature_log_prob_.at(c);
          for (int j = 0; j < X.cols(); ++j) {
            double x = X(i, j);
            if (x > 0)
              log_prob += x * log_prob_c[j];
          }
          if (log_prob > best_log_prob) {
            best_log_prob = log_prob;
            best_class = c;
          }
        }
        preds[i] = best_class;
      }
      return preds;
    }
  };

  /** \brief Bernoulli Naive Bayes Classifier.
   */
  class BernoulliNB {
  public:
    double alpha;
    std::vector<int> classes_;
    std::map<int, double> class_prior_;
    std::map<int, Eigen::VectorXd> feature_log_prob_;
    std::map<int, Eigen::VectorXd> feature_log_prob_neg_;

    BernoulliNB(double alpha = 1.0) : alpha(alpha) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXi &y) {
      classes_.clear();
      class_prior_.clear();
      feature_log_prob_.clear();
      feature_log_prob_neg_.clear();
      int n = X.rows(), m = X.cols();
      std::map<int, int> count;
      for (int i = 0; i < n; ++i)
        count[y[i]]++;
      for (auto &p : count)
        classes_.push_back(p.first);
      for (int c : classes_) {
        class_prior_[c] = (double)count[c] / n;
        Eigen::VectorXd count1 = Eigen::VectorXd::Zero(m);
        int n_c = count[c];
        for (int i = 0; i < n; ++i) {
          if (y[i] == c)
            count1 += X.row(i).transpose();
        }
        Eigen::VectorXd log_prob1(m);
        Eigen::VectorXd log_prob0(m);
        for (int j = 0; j < m; ++j) {
          double count1_j = count1[j];
          double p1 = (count1_j + alpha) / (n_c + 2 * alpha);
          double p0 = 1 - p1;
          log_prob1[j] = std::log(p1);
          log_prob0[j] = std::log(p0);
        }
        feature_log_prob_[c] = log_prob1;
        feature_log_prob_neg_[c] = log_prob0;
      }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi preds(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double best_log_prob = -1e9;
        int best_class = classes_[0];
        for (int c : classes_) {
          double log_prob = std::log(class_prior_.at(c));
          const Eigen::VectorXd &log_prob1 = feature_log_prob_.at(c);
          const Eigen::VectorXd &log_prob0 = feature_log_prob_neg_.at(c);
          for (int j = 0; j < X.cols(); ++j) {
            if (X(i, j) > 0.5)
              log_prob += log_prob1[j];
            else
              log_prob += log_prob0[j];
          }
          if (log_prob > best_log_prob) {
            best_log_prob = log_prob;
            best_class = c;
          }
        }
        preds[i] = best_class;
      }
      return preds;
    }
  };

} // namespace naive_bayes

//////////////////////////////
// Clustering
//////////////////////////////
namespace cluster {

  /** \brief K-Means clustering algorithm.
   */
  class KMeans {
  public:
    int n_clusters;
    int max_iter;
    Eigen::MatrixXd centroids;
    Eigen::VectorXi labels_;
    
    KMeans(int n_clusters = 8, int max_iter = 300)
      : n_clusters(n_clusters), max_iter(max_iter) {}

    void fit(const Eigen::MatrixXd &X) {
      int n_samples = X.rows();
      int n_features = X.cols();
      labels_ = Eigen::VectorXi::Zero(n_samples);
      centroids = Eigen::MatrixXd::Zero(n_clusters, n_features);
      for (int k = 0; k < n_clusters; ++k)
        if (k < n_samples)
          centroids.row(k) = X.row(k);
      for (int iter = 0; iter < max_iter; ++iter) {
        bool changed = false;
        for (int i = 0; i < n_samples; ++i) {
          double best_dist = std::numeric_limits<double>::infinity();
          int best_cluster = 0;
          for (int k = 0; k < n_clusters; ++k) {
            double dist = (X.row(i) - centroids.row(k)).squaredNorm();
            if (dist < best_dist) {
              best_dist = dist;
              best_cluster = k;
            }
          }
          if (labels_[i] != best_cluster) {
            labels_[i] = best_cluster;
            changed = true;
          }
        }
        Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(n_clusters, n_features);
        Eigen::VectorXi count = Eigen::VectorXi::Zero(n_clusters);
        for (int i = 0; i < n_samples; ++i) {
          new_centroids.row(labels_[i]) += X.row(i);
          count[labels_[i]] += 1;
        }
        for (int k = 0; k < n_clusters; ++k) {
          if (count[k] > 0)
            new_centroids.row(k) /= count[k];
          else
            new_centroids.row(k) = X.row(rand() % n_samples);
        }
        centroids = new_centroids;
        if (!changed)
          break;
      }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd &X) const {
      Eigen::VectorXi labels(X.rows());
      for (int i = 0; i < X.rows(); ++i) {
        double best_dist = std::numeric_limits<double>::infinity();
        int best_cluster = 0;
        for (int k = 0; k < n_clusters; ++k) {
          double dist = (X.row(i) - centroids.row(k)).squaredNorm();
          if (dist < best_dist) {
            best_dist = dist;
            best_cluster = k;
          }
        }
        labels[i] = best_cluster;
      }
      return labels;
    }
  };

  /** \brief DBSCAN clustering algorithm.
   */
  class DBSCAN {
  public:
    double eps;
    int min_samples;
    Eigen::VectorXi labels_;
    
    DBSCAN(double eps = 0.5, int min_samples = 5)
      : eps(eps), min_samples(min_samples) {}

    void fit(const Eigen::MatrixXd &X) {
      int n = X.rows();
      labels_ = Eigen::VectorXi::Constant(n, -1);
      std::vector<bool> visited(n, false);
      int cluster_id = 0;
      for (int i = 0; i < n; ++i) {
        if (visited[i])
          continue;
        visited[i] = true;
        std::vector<int> neighbors = region_query(X, i);
        if (neighbors.size() < (size_t)min_samples) {
          labels_[i] = -1;
        } else {
          labels_[i] = cluster_id;
          for (size_t j = 0; j < neighbors.size(); ++j) {
            int idx = neighbors[j];
            if (!visited[idx]) {
              visited[idx] = true;
              std::vector<int> neighbors2 = region_query(X, idx);
              if (neighbors2.size() >= (size_t)min_samples) {
                neighbors.insert(neighbors.end(), neighbors2.begin(), neighbors2.end());
              }
            }
            if (labels_[idx] == -1)
              labels_[idx] = cluster_id;
            if (labels_[idx] == -1 || labels_[idx] == -2)
              labels_[idx] = cluster_id;
          }
          cluster_id++;
        }
      }
    }

  private:
    std::vector<int> region_query(const Eigen::MatrixXd &X, int i) const {
      std::vector<int> neighbors;
      for (int j = 0; j < X.rows(); ++j) {
        double dist = (X.row(i) - X.row(j)).norm();
        if (dist <= eps)
          neighbors.push_back(j);
      }
      return neighbors;
    }
  };

  /** \brief Agglomerative (Hierarchical) Clustering.
   */
  class AgglomerativeClustering {
  public:
    int n_clusters;
    Eigen::VectorXi labels_;
    
    AgglomerativeClustering(int n_clusters = 2)
      : n_clusters(n_clusters) {}

    void fit(const Eigen::MatrixXd &X) {
      int n = X.rows();
      labels_ = Eigen::VectorXi(n);
      for (int i = 0; i < n; ++i)
        labels_[i] = i;
      int current_clusters = n;
      Eigen::MatrixXd dist = Eigen::MatrixXd::Zero(n, n);
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
          dist(i, j) = dist(j, i) = (X.row(i) - X.row(j)).norm();
      while (current_clusters > n_clusters) {
        int ci = -1, cj = -1;
        double best_dist = 1e9;
        for (int a = 0; a < n; ++a) {
          for (int b = a + 1; b < n; ++b) {
            if (labels_[a] == labels_[b])
              continue;
            if (dist(a, b) < best_dist) {
              best_dist = dist(a, b);
              ci = labels_[a];
              cj = labels_[b];
            }
          }
        }
        for (int k = 0; k < n; ++k) {
          if (labels_[k] == cj)
            labels_[k] = ci;
        }
        current_clusters--;
      }
      std::map<int, int> remap;
      int new_label = 0;
      for (int i = 0; i < n; ++i) {
        if (remap.find(labels_[i]) == remap.end()) {
          remap[labels_[i]] = new_label++;
        }
        labels_[i] = remap[labels_[i]];
      }
    }
  };

} // namespace cluster

//////////////////////////////
// Dimensionality Reduction
//////////////////////////////
namespace decomposition {

  /** \brief Principal Component Analysis (PCA).
   */
  class PCA {
  public:
    int n_components;
    Eigen::MatrixXd components_;
    Eigen::VectorXd explained_variance_;
    Eigen::VectorXd mean_;
    
    PCA(int n_components = 2) : n_components(n_components) {}

    void fit(const Eigen::MatrixXd &X) {
      mean_ = X.colwise().mean();
      Eigen::MatrixXd X_centered = X.rowwise() - mean_.transpose();
      Eigen::MatrixXd Cov = (X_centered.adjoint() * X_centered) / double(X.rows() - 1);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Cov);
      Eigen::VectorXd eigvals = solver.eigenvalues();
      Eigen::MatrixXd eigvecs = solver.eigenvectors();
      std::vector<int> indices(eigvals.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(), [&](int a, int b) { return eigvals[a] > eigvals[b]; });
      int k = std::min(n_components, (int)eigvals.size());
      components_ = Eigen::MatrixXd(k, X.cols());
      explained_variance_ = Eigen::VectorXd(k);
      for (int i = 0; i < k; ++i) {
        components_.row(i) = eigvecs.col(indices[i]).transpose();
        explained_variance_[i] = eigvals[indices[i]];
      }
    }

    Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const {
      Eigen::MatrixXd X_centered = X.rowwise() - mean_.transpose();
      Eigen::MatrixXd X_reduced(X.rows(), components_.rows());
      for (int i = 0; i < components_.rows(); ++i)
        X_reduced.col(i) = X_centered * components_.row(i).transpose();
      return X_reduced;
    }
  };

  /** \brief Truncated SVD.
   */
  class TruncatedSVD {
  public:
    int n_components;
    Eigen::MatrixXd components_;
    Eigen::VectorXd explained_variance_;
    
    TruncatedSVD(int n_components = 2) : n_components(n_components) {}

    void fit(const Eigen::MatrixXd &X) {
      Eigen::BDCSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
      int k = std::min(n_components, (int)X.cols());
      components_ = svd.matrixV().leftCols(k).transpose();
      explained_variance_ = svd.singularValues().head(k).array().square();
    }

    Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const {
      return X * components_.transpose();
    }
  };

  /** \brief Non-Negative Matrix Factorization (NMF).
   */
  class NMF {
  public:
    int n_components;
    int max_iter;
    Eigen::MatrixXd W;
    Eigen::MatrixXd H;
    
    NMF(int n_components = 2, int max_iter = 200) : n_components(n_components), max_iter(max_iter) {}

    Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X) {
      int n = X.rows(), m = X.cols();
      W = Eigen::MatrixXd::Random(n, n_components).cwiseAbs();
      H = Eigen::MatrixXd::Random(n_components, m).cwiseAbs();
      for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd WH = W * H;
        Eigen::MatrixXd numeratorW = X * H.transpose();
        Eigen::MatrixXd denominatorW = WH * H.transpose();
        denominatorW.array() += 1e-9;
        W = W.cwiseProduct(numeratorW.cwiseQuotient(denominatorW));
        WH = W * H;
        Eigen::MatrixXd numeratorH = W.transpose() * X;
        Eigen::MatrixXd denominatorH = W.transpose() * WH;
        denominatorH.array() += 1e-9;
        H = H.cwiseProduct(numeratorH.cwiseQuotient(denominatorH));
      }
      return W;
    }
  };

} // namespace decomposition

//////////////////////////////
// Manifold Learning
//////////////////////////////
namespace manifold {

  /** \brief t-Distributed Stochastic Neighbor Embedding (t-SNE).
   */
  class TSNE {
  public:
    int n_components;
    double perplexity;
    double learning_rate;
    int n_iter;
    
    TSNE(int n_components = 2, double perplexity = 30.0, double learning_rate = 200.0, int n_iter = 1000)
      : n_components(n_components), perplexity(perplexity), learning_rate(learning_rate), n_iter(n_iter) {}

    Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X) {
      int n = X.rows();
      Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n, n_components);
      return Y;
    }
  };

  /** \brief Isomap.
   */
  class Isomap {
  public:
    int n_components;
    int n_neighbors;
    
    Isomap(int n_components = 2, int n_neighbors = 5)
      : n_components(n_components), n_neighbors(n_neighbors) {}

    Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X) {
      Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(X.rows(), n_components);
      return Y;
    }
  };

} // namespace manifold

//////////////////////////////
// Preprocessing Utilities
//////////////////////////////
namespace preprocessing {

  /** \brief StandardScaler: removes mean and scales to unit variance.
   */
  class StandardScaler {
  public:
    Eigen::RowVectorXd mean_;
    Eigen::RowVectorXd scale_;
    
    void fit(const Eigen::MatrixXd &X) {
      mean_ = X.colwise().mean();
      Eigen::RowVectorXd var = Eigen::RowVectorXd::Zero(X.cols());
      for (int j = 0; j < X.cols(); ++j)
        var[j] = (X.col(j).array() - mean_[j]).square().sum() / X.rows();
      scale_ = var.array().sqrt();
      for (int j = 0; j < scale_.size(); ++j)
        if (scale_[j] == 0)
          scale_[j] = 1;
    }

    Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const {
      Eigen::MatrixXd X_tr = X.rowwise() - mean_;
      for (int j = 0; j < X.cols(); ++j)
        X_tr.col(j) = X_tr.col(j) / scale_[j];
      return X_tr;
    }

    Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X) {
      fit(X);
      return transform(X);
    }
  };

  /** \brief MinMaxScaler: scales features to a given range.
   */
  class MinMaxScaler {
  public:
    Eigen::RowVectorXd min_;
    Eigen::RowVectorXd max_;
    double feature_min;
    double feature_max;

    MinMaxScaler(double feature_min = 0.0, double feature_max = 1.0)
      : feature_min(feature_min), feature_max(feature_max) {}

    void fit(const Eigen::MatrixXd &X) {
      min_ = X.colwise().minCoeff();
      max_ = X.colwise().maxCoeff();
      for (int j = 0; j < min_.size(); ++j)
        if (max_[j] - min_[j] == 0)
          max_[j] = min_[j] + 1e-9;
    }

    Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const {
      Eigen::MatrixXd X_tr = Eigen::MatrixXd::Zero(X.rows(), X.cols());
      for (int j = 0; j < X.cols(); ++j)
        X_tr.col(j) = ((X.col(j).array() - min_[j]) / (max_[j] - min_[j])) * (feature_max - feature_min) + feature_min;
      return X_tr;
    }

    Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X) {
      fit(X);
      return transform(X);
    }
  };

  /** \brief OneHotEncoder: encodes categorical features as one-hot vectors.
   */
  class OneHotEncoder {
  public:
    std::vector<std::vector<int>> categories_;

    void fit(const Eigen::MatrixXi &X) {
      categories_.clear();
      categories_.resize(X.cols());
      for (int j = 0; j < X.cols(); ++j) {
        std::set<int> cats;
        for (int i = 0; i < X.rows(); ++i)
          cats.insert(X(i, j));
        categories_[j] = std::vector<int>(cats.begin(), cats.end());
        std::sort(categories_[j].begin(), categories_[j].end());
      }
    }

    Eigen::MatrixXd transform(const Eigen::MatrixXi &X) const {
      int total_dims = 0;
      std::vector<int> offsets;
      for (const auto &cat_list : categories_) {
        offsets.push_back(total_dims);
        total_dims += cat_list.size();
      }
      Eigen::MatrixXd X_out = Eigen::MatrixXd::Zero(X.rows(), total_dims);
      for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
          int val = X(i, j);
          auto it = std::find(categories_[j].begin(), categories_[j].end(), val);
          if (it != categories_[j].end()) {
            int idx = it - categories_[j].begin();
            X_out(i, offsets[j] + idx) = 1.0;
          }
        }
      }
      return X_out;
    }

    Eigen::MatrixXd fit_transform(const Eigen::MatrixXi &X) {
      fit(X);
      return transform(X);
    }
  };

  /** \brief LabelEncoder: encodes labels as integers.
   */
  template<typename T>
  class LabelEncoder {
  public:
    std::map<T, int> label_to_index_;
    std::vector<T> classes_;

    void fit(const std::vector<T> &y) {
      label_to_index_.clear();
      classes_.clear();
      for (const T &label : y) {
        if (label_to_index_.find(label) == label_to_index_.end()) {
          int index = label_to_index_.size();
          label_to_index_[label] = index;
          classes_.push_back(label);
        }
      }
    }

    std::vector<int> transform(const std::vector<T> &y) const {
      std::vector<int> result;
      result.reserve(y.size());
      for (const T &label : y) {
        auto it = label_to_index_.find(label);
        if (it == label_to_index_.end())
          throw std::runtime_error("LabelEncoder: unknown label in transform");
        result.push_back(it->second);
      }
      return result;
    }

    std::vector<T> inverse_transform(const std::vector<int> &y_idx) const {
      std::vector<T> result;
      result.reserve(y_idx.size());
      for (int idx : y_idx) {
        if (idx < 0 || idx >= (int)classes_.size())
          throw std::runtime_error("LabelEncoder: index out of range in inverse_transform");
        result.push_back(classes_[idx]);
      }
      return result;
    }
  };

  /** \brief SimpleImputer: fills in missing values.
   */
  class SimpleImputer {
  public:
    std::string strategy;
    double fill_value;
    Eigen::RowVectorXd statistics_;
    
    SimpleImputer(std::string strategy = "mean", double fill_value = 0.0)
      : strategy(strategy), fill_value(fill_value) {}

    void fit(const Eigen::MatrixXd &X) {
      statistics_ = Eigen::RowVectorXd::Zero(X.cols());
      if (strategy == "mean") {
        for (int j = 0; j < X.cols(); ++j) {
          double sum = 0.0;
          int count = 0;
          for (int i = 0; i < X.rows(); ++i) {
            double val = X(i, j);
            if (std::isfinite(val)) { sum += val; count++; }
          }
          statistics_[j] = (count > 0 ? sum / count : NAN);
        }
      } else if (strategy == "median") {
        for (int j = 0; j < X.cols(); ++j) {
          std::vector<double> vals;
          for (int i = 0; i < X.rows(); ++i) {
            if (std::isfinite(X(i, j)))
              vals.push_back(X(i, j));
          }
          if (!vals.empty()) {
            std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
            statistics_[j] = vals[vals.size() / 2];
          } else {
            statistics_[j] = NAN;
          }
        }
      } else if (strategy == "most_frequent") {
        for (int j = 0; j < X.cols(); ++j) {
          std::map<double, int> freq;
          for (int i = 0; i < X.rows(); ++i) {
            double val = X(i, j);
            if (std::isfinite(val))
              freq[val] += 1;
          }
          int max_count = -1;
          double mode = NAN;
          for (auto &p : freq) {
            if (p.second > max_count) { max_count = p.second; mode = p.first; }
          }
          statistics_[j] = mode;
        }
      } else if (strategy == "constant") {
        for (int j = 0; j < X.cols(); ++j)
          statistics_[j] = fill_value;
      }
    }

    Eigen::MatrixXd transform(Eigen::MatrixXd X) const {
      for (int j = 0; j < X.cols(); ++j) {
        double fill = statistics_[j];
        for (int i = 0; i < X.rows(); ++i) {
          if (!std::isfinite(X(i, j)))
            X(i, j) = fill;
        }
      }
      return X;
    }

    Eigen::MatrixXd fit_transform(Eigen::MatrixXd X) {
      fit(X);
      return transform(X);
    }
  };

  /** \brief PolynomialFeatures: generates polynomial combinations of features.
   */
  class PolynomialFeatures {
  public:
    int degree;
    bool include_bias;
    int n_input_features_;
    int n_output_features_;

    PolynomialFeatures(int degree = 2, bool include_bias = true)
      : degree(degree), include_bias(include_bias), n_input_features_(0), n_output_features_(0) {}

    void fit(const Eigen::MatrixXd &X) {
      n_input_features_ = X.cols();
      n_output_features_ = 0;
      for (int d = 0; d <= degree; ++d) {
        if (d == 0) {
          if (include_bias)
            n_output_features_ += 1;
        } else {
          long num = 1;
          for (int i = 1; i <= d; ++i)
            num = num * (n_input_features_ + i - 1) / i;
          n_output_features_ += num;
        }
      }
      if (!include_bias)
        n_output_features_ -= 1;
    }

    Eigen::MatrixXd transform(const Eigen::MatrixXd &X) const {
      int n = X.rows();
      if (degree == 2) {
        int p = n_input_features_;
        int output_cols = include_bias ? 1 + p + p * (p + 1) / 2 : p + p * (p - 1) / 2;
        Eigen::MatrixXd X_poly(n, output_cols);
        int col = 0;
        if (include_bias) {
          X_poly.col(col++).setOnes();
        }
        for (int j = 0; j < p; ++j) {
          X_poly.col(col++) = X.col(j);
        }
        for (int j = 0; j < p; ++j) {
          for (int k = j; k < p; ++k) {
            X_poly.col(col++) = X.col(j).cwiseProduct(X.col(k));
          }
        }
        return X_poly;
      } else {
        Eigen::MatrixXd X_poly = X;
        if (include_bias) {
          Eigen::MatrixXd X_poly2(n, X.cols() + 1);
          X_poly2 << Eigen::VectorXd::Ones(n), X;
          X_poly = X_poly2;
        }
        return X_poly;
      }
    }

    Eigen::MatrixXd fit_transform(const Eigen::MatrixXd &X) {
      fit(X);
      return transform(X);
    }
  };

} // namespace preprocessing

//////////////////////////////
// Pipeline Tools
//////////////////////////////
namespace pipeline {

  /** \brief Pipeline: chains transformers and an estimator.
   */
  class Pipeline {
  private:
    struct StepBase {
      virtual ~StepBase() = default;
      virtual void fit(Eigen::MatrixXd &X, Eigen::VectorXd &y) = 0;
      virtual Eigen::MatrixXd transform(const Eigen::MatrixXd &X) { return X; }
      virtual Eigen::VectorXd predict(const Eigen::MatrixXd &X) { return Eigen::VectorXd(); }
    };

    template<typename Transformer>
    struct TransformerStep : StepBase {
      Transformer transformer;
      explicit TransformerStep(const Transformer &trans) : transformer(trans) {}
      void fit(Eigen::MatrixXd &X, Eigen::VectorXd &y) override {
        transformer.fit(X);
        Eigen::MatrixXd X_out = transformer.transform(X);
        X = X_out;
      }
      Eigen::MatrixXd transform(const Eigen::MatrixXd &X) override {
        return transformer.transform(X);
      }
    };

    template<typename Estimator>
    struct EstimatorStep : StepBase {
      Estimator estimator;
      explicit EstimatorStep(const Estimator &est) : estimator(est) {}
      void fit(Eigen::MatrixXd &X, Eigen::VectorXd &y) override {
        estimator.fit(X, y);
      }
      Eigen::VectorXd predict(const Eigen::MatrixXd &X) override {
        auto result = estimator.predict(X);
        Eigen::VectorXd output;
        if constexpr (std::is_same_v<decltype(result), Eigen::VectorXi>)
          output = result.template cast<double>();
        else
          output = result;
        return output;
      }
    };

    std::vector<std::string> step_names_;
    std::vector<std::unique_ptr<StepBase>> steps_;
  
  public:
    template<typename Transformer>
    void add_transformer(const std::string &name, const Transformer &trans) {
      step_names_.push_back(name);
      steps_.emplace_back(new TransformerStep<Transformer>(trans));
    }
    template<typename Estimator>
    void add_estimator(const std::string &name, const Estimator &est) {
      step_names_.push_back(name);
      steps_.emplace_back(new EstimatorStep<Estimator>(est));
    }

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
      Eigen::MatrixXd X_temp = X;
      Eigen::VectorXd y_temp = y;
      for (size_t i = 0; i < steps_.size(); ++i) {
        bool last_step = (i == steps_.size() - 1);
        steps_[i]->fit(X_temp, y_temp);
      }
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const {
      Eigen::MatrixXd X_temp = X;
      for (size_t i = 0; i < steps_.size(); ++i) {
        bool last_step = (i == steps_.size() - 1);
        if (last_step)
          return steps_[i]->predict(X_temp);
        else
          X_temp = steps_[i]->transform(X_temp);
      }
      return Eigen::VectorXd();
    }
  };

} // namespace pipeline

//////////////////////////////
// Model Selection
//////////////////////////////
namespace model_selection {

  /** \brief Splits X and y into training and testing sets.
   */
  inline std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::pair<Eigen::VectorXd, Eigen::VectorXd>>
  train_test_split(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, double test_size = 0.25, unsigned int random_state = 0) {
    int n = X.rows();
    int n_test = static_cast<int>(n * test_size);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(random_state);
    std::shuffle(indices.begin(), indices.end(), gen);
    std::vector<int> test_idx(indices.begin(), indices.begin() + n_test);
    std::vector<int> train_idx(indices.begin() + n_test, indices.end());
    int n_train = train_idx.size();
    Eigen::MatrixXd X_train(n_train, X.cols());
    Eigen::MatrixXd X_test(n_test, X.cols());
    Eigen::VectorXd y_train(n_train);
    Eigen::VectorXd y_test(n_test);
    for (int i = 0; i < n_train; ++i) {
      X_train.row(i) = X.row(train_idx[i]);
      y_train[i] = y[train_idx[i]];
    }
    for (int j = 0; j < n_test; ++j) {
      X_test.row(j) = X.row(test_idx[j]);
      y_test[j] = y[test_idx[j]];
    }
    return { { X_train, X_test }, { y_train, y_test } };
  }

  template<typename Estimator>
  std::vector<double> cross_val_score(Estimator estimator, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int cv = 5) {
    int n = X.rows();
    std::vector<double> scores;
    int fold_size = n / cv;
    for (int i = 0; i < cv; ++i) {
      int start = i * fold_size;
      int end = (i == cv - 1) ? n : start + fold_size;
      Eigen::MatrixXd X_val = X.middleRows(start, end - start);
      Eigen::VectorXd y_val = y.segment(start, end - start);
      Eigen::MatrixXd X_train(X.rows() - (end - start), X.cols());
      Eigen::VectorXd y_train(y.size() - (end - start));
      int idx = 0;
      for (int j = 0; j < n; ++j) {
        if (j >= start && j < end)
          continue;
        X_train.row(idx) = X.row(j);
        y_train[idx] = y[j];
        idx++;
      }
      estimator.fit(X_train, y_train);
      Eigen::VectorXd y_pred = estimator.predict(X_val);
      double score = 0.0;
      if (y_val.size() > 0) {
        bool classification = true;
        for (int k = 0; k < y_val.size(); ++k) {
          if (std::floor(y_val[k]) != y_val[k]) { classification = false; break; }
        }
        if (classification) {
          int correct = 0;
          for (int k = 0; k < y_val.size(); ++k) {
            if (std::round(y_pred[k]) == y_val[k])
              correct++;
          }
          score = (double)correct / y_val.size();
        } else {
          double y_mean = y_val.mean();
          double ss_tot = 0.0, ss_res = 0.0;
          for (int k = 0; k < y_val.size(); ++k) {
            ss_tot += (y_val[k] - y_mean) * (y_val[k] - y_mean);
            ss_res += (y_val[k] - y_pred[k]) * (y_val[k] - y_pred[k]);
          }
          score = (ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0);
        }
      }
      scores.push_back(score);
    }
    return scores;
  }

  template<typename Estimator>
  class GridSearchCV {
  public:
    Estimator best_estimator_;
    std::map<std::string, double> best_params_;
    double best_score_;
    std::vector<std::map<std::string, double>> param_grid_;
    int cv;

    GridSearchCV(const std::vector<std::map<std::string, double>> &param_grid, int cv = 5)
      : best_score_(-std::numeric_limits<double>::infinity()), param_grid_(param_grid), cv(cv) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const Estimator &estimator_prototype) {
      best_score_ = -std::numeric_limits<double>::infinity();
      best_params_.clear();
      for (const auto &params : param_grid_) {
        Estimator model = estimator_prototype;
        for (const auto &kv : params) {
          const std::string &param = kv.first;
          double value = kv.second;
          // Pseudocode: set parameter if estimator supports it.
          // e.g., if (param == "alpha") model.alpha = value;
        }
        std::vector<double> scores = cross_val_score(model, X, y, cv);
        double mean_score = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        if (mean_score > best_score_) {
          best_score_ = mean_score;
          best_params_ = params;
          best_estimator_ = model;
          best_estimator_.fit(X, y);
        }
      }
    }
  };

} // namespace model_selection

} // namespace neurite

#endif // NEURITE_HPP
