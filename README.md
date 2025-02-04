# MLOps-890
This is a repository for the MLOps course at MSU for spring 2025
## Model Comparison: Single vs. Two-Feature Linear Regression

Initially, a **single-feature linear regression model** was built using the predictor that showed the highest correlation with the target variable (`y`). This model provided a simple and interpretable relationship between the x4 and `y`. However, after evaluating multiple predictors, we extended the model to include **two predictors** (`x2` and `x4`), which demonstrated better performance. The **two-feature model** improved accuracy by capturing more variance in `y`, as measured by a higher R² score.The single-feature model is still accessible for reference, ensuring version control allows for comparison and rollback if necessary.
