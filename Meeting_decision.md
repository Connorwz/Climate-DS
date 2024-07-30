# 07/02/2024
1. Replace neural networks in paper to models such as random forest and XGB.
2. Visualize true and predicted $g(\sigma)$.

# 07/09/2024
1. For random forest: be careful about data/feature ratio used to fit the model; consider output in a vector; look at parameters in R.
2. Would like to see mse/correlation/rank correlation; produce a table to find those having high/low correlation and high/low mse.
3. Think about inter-independence between nodes.
4. Find samples that make most mistakes (eg. L1 between g(sigma) and g-hat(sigma))/(eg. outliers that will influence correlation coefficient)
5. visualize where's the biggest error and why error happens
6. Notice the pattern and choose the tool to see: 1.Node 11 and 12: shift, 2.Node 7,8:model complexity, 3.NodeNode11-13: model diverges for different combinations, detect continuity, 4.
7. Xgboost: use early stopping, unite model contunity.
8. Use cubic smoothing splines, weighted combination of splines (more details on whiteboard)
9. Feature function regression
10. Customize loss function, then find g-hat(sigma), then compare it with g(sigma). 

# 07/16/2024
1. How epoch in xgboost? Since in boosting weight is changed by previous round.
2. 2*2 plots: draw for each nodes, for xgboost and neural network
3. Plot mae, cc for variable, marginal dependence plot between curve and nodes
4. learn how xgboost works and discuss
