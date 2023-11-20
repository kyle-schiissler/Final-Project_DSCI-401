set.seed(300344635)
#### Data Cleaning 

train_data <- read.csv("train.csv")

train_data <- train_data[ , -which(names(train_data) %in% c("uid","stand", "inning", "inning_topbot", "if_fielding_alignment", "of_fielding_alignment", "on_3b", "on_2b", "on_1b"))]

# List of columns to convert to factors
cols_to_factor <- c("pitch_type", "p_throws")

train_data <- train_data %>%
  mutate_at(vars(cols_to_factor), as.factor)

# REDUCING SAMPLE TO WORK WITH 

df <- train_data[0:10000, ]

##NEEDS TO BE TUNNED PROPERLY

##### XGBoost 
# Prepare the data
xgb_prep <- recipe(is_strike ~ ., -is_strike, data = df) %>% 
  step_integer(all_nominal()) %>% 
  prep(training = df, retain = TRUE) %>% 
  juice() 
X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "is_strike")]) 
Y <- as.integer(df$is_strike)

is.factor(Y)

# Cross-validation with XGBoost
df_xgb <- xgboost::xgb.cv(
  data = X,
  label = Y,
  nrounds = 6000,
  objective = "binary:logistic", # Assuming is_strike is a binary classification target
  early_stopping_rounds = 50, 
  nfold = 10,
  params = list(
    eta = 0.1,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.5,
    colsample_bytree = 1.0, 
    n.minobsinnode = 10,
    eval_metric = "logloss"),
  verbose = 0
)  

# Find index for number trees with minimum CV error
best_nrounds <- which.min(df_xgb$evaluation_log$test_logloss_mean)

# Get the best Log Loss
best_log_loss <- df_xgb$evaluation_log$test_logloss_mean[best_nrounds]

# Print the best Log Loss
print(paste("Best Log Loss:", best_log_loss))

# Optionally, visualize the performance over boosting rounds
# Plotting log loss over rounds
library(ggplot2)
logloss_data <- data.frame(round = 1:nrow(df_xgb$evaluation_log), logloss = df_xgb$evaluation_log$test_logloss_mean)
ggplot(logloss_data, aes(x = round, y = logloss)) +
  geom_line() +
  geom_vline(xintercept = best_nrounds, linetype = "dashed", color = "red") +
  labs(title = "Log Loss over Boosting Rounds", x = "Boosting Round", y = "Log Loss")

# Get performance
gbm.perf(df_xgb, method = "cv") 

#Loss functions: XGBoost allows users to define and optimize gradient boosting models using custom objective and evaluation criteria.
#eval on log loss 