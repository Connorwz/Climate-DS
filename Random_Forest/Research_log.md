# 7/2/2024
1. It's found that random forest models cannot be fitted without GPU-acceleration: choose cuML to accelerate model fitting;
2. cuML supports only MSE as split criterion and Y $\in \mathbb{R}$;
3. Use GridSearchCV to find best hyperparameters where accuracy metric is MAE for each node;
4. Linear correlation coefficient is 0.11: it's found that preprocessing codes have problems;
5. Fix the problem and linear correlation coefficient is 0.32;
6. Use GridSearchCV to find best hyperparameters where accuracy metric is MSE for each node; It's found that more estimators and fewer depth are favored so modified the range of hyperparameters tuned;
7. Linear correlation coefficient is 0.42;
