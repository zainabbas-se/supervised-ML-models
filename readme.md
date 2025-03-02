# Machine Learning Models Implementation

This repository contains implementations of various machine learning algorithms from scratch using Python. The project includes both classification and regression models, demonstrating fundamental machine learning concepts through practical examples.

## Models Implemented

### Classification Models
1. **Logistic Regression**
   - Custom implementation for binary classification
   - Uses Gradient Descent with Binary Cross-Entropy loss
   - Applied to diabetes prediction dataset

2. **K-Nearest Neighbors (KNN) Classifier**
   - Custom implementation with support for multiple distance metrics
   - Includes both Euclidean and Manhattan distance calculations
   - Compared with scikit-learn's KNN implementation

3. **Support Vector Machine (SVM)**
   - Custom implementation using Hinge Loss
   - Includes regularization with lambda parameter
   - Applied to binary classification problems

### Regression Models
1. **Linear Regression**
   - Custom implementation using Gradient Descent
   - Uses Mean Squared Error (MSE) loss function
   - Applied to salary prediction dataset

2. **KNN Regression**
   - Implementation for continuous value prediction
   - Uses nearest neighbor averaging for predictions

## Dataset Used

1. **Diabetes Dataset**
   - Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
   - Target: Outcome (Binary: 0 or 1)
   - Used for classification models (Logistic Regression, KNN, SVM)

2. **Salary Dataset**
   - Used for regression analysis
   - Applied with Linear Regression model

## Implementation Details

### Common Features
- All models implement `fit()` and `predict()` methods
- Custom implementations without using scikit-learn's model classes
- Includes data preprocessing and train-test splitting
- Performance evaluation using appropriate metrics

### Hyperparameters
- Learning rates
- Number of iterations
- K values for KNN
- Lambda parameter for SVM regularization

## Requirements
- Python 3.x
- NumPy (>= 1.18.0)
- Pandas (>= 1.0.0)
- Scikit-learn (for comparison and data splitting, >= 0.22.0)
- Matplotlib (for visualization, >= 3.0.0)


## Future Improvements
1. Add more distance metrics for KNN
2. Implement k-fold cross-validation
3. Add feature scaling options
4. Include more performance metrics
5. Add visualization tools for decision boundaries

## Contributing
Feel free to contribute to this project by:
1. Implementing new algorithms
2. Adding more datasets
3. Improving existing implementations
4. Adding documentation
5. Fixing bugs

## License
This project is open source and available under the MIT License.