require(tidyverse)
require(dplyr)
require(tidyr)
require(magrittr)
require(caret)
require(vip)
library(ggplot2)     # for awesome plotting
library(rpart)       # direct engine for decision tree application
require(ranger)      # Random Forest   
library(rpart.plot)  # for plotting decision trees
library(pdp)         # for feature effects
require(recipes)
require(xgboost)

set.seed(300344635)
#### Data Cleaning 

train_data <- read.csv("train.csv")

train_data <- train_data[ , -which(names(train_data) %in% c("uid","stand", "inning", "inning_topbot", "if_fielding_alignment", "of_fielding_alignment", "on_3b", "on_2b", "on_1b"))]

# List of columns to convert to factors
cols_to_factor <- c("pitch_type", "p_throws")

train_data <- train_data %>%
  mutate_at(vars(cols_to_factor), as.factor)

# Convert is_strike to a factor 
train_data$is_strike <- as.factor(train_data$is_strike)

is.factor(train_data$is_strike)

# REDUCING SAMPLE TO WORK WITH 

df <- train_data[0:10000, ]

## Find best logistic regression model 

log_reg_model <- glm(
  is_strike ~ ., 
  family = "binomial",
  data = df
)


# Coefficients 
exp(coef(log_reg_model))

# Log odds 
confint(log_reg_model)

cv_log_reg_model <- caret::train(is_strike ~ .,
                     data = df, 
                     method = "glm",
                     family = "binomial",
                     trControl = trainControl(method = "cv", number = 10)
                     )

cv_log_reg_model


# https://www.xlstat.com/en/solutions/features/partial-least-squares-regression#:~:text=The%20Partial%20Least%20Squares%20regression,used%20to%20perfom%20a%20regression.

#What is the difference between PCR and PLS regression?
#The components obtained from the PLS regression,which is based on covariance, are built so that they explain as well as possible Y, while the components of the PCR are built to describe X as well as possible. This explains why the PLS regression outperforms PCR when the target is strongly correlated with a direction in the data that have a low variance. The XLSTAT-PLS software allows partly compensating this drawback of the PCR by allowing the selection of the components that are the most correlated with Y.
log_reg_pls_model <- caret::train(is_strike ~ .,
                                  data = df, 
                                  method = "glm", 
                                  family = "binomial", 
                                  trControl = caret::trainControl(method = "cv", number = 5),
                                  preProcess = c("zv", "center", "scale"), # note: method = "zv" identifies numeric predictor columns with a single value (i.e. having zero variance) and excludes them from further calculations.

                                  tuneLength = 26)

log_reg_pls_model


# Pre processing for glmnet 

# Create training  feature matrices
# we use model.matrix(...)[, -1] to discard the intercept
X <- model.matrix(is_strike ~ ., df)[, -1]

Y <- (df$is_strike)

is.factor(Y)

log_reg_regularize_model <- caret::train(
  x = X,
  y = Y,
  method = "glmnet",
  family = "binomial",
  preProc = c("zv", "center", "scale"),
  trControl = caret::trainControl(method = "cv", number = 5),
  tuneLength = 15
)

log_reg_regularize_model


log_reg_regularize_model$results %>%
  filter(alpha == log_reg_regularize_model$bestTune$alpha, lambda == log_reg_regularize_model$bestTune$lambda)

ggplot(log_reg_regularize_model)

vip(log_reg_regularize_model, num_features = 17, geom = "point")

##### Random Forest section 
n_features <- length(setdiff(names(df), "is_strike"))

hyper_grid <- expand.grid(
  mtry = floor(sqrt(n_features) * c(.5, .75, 1.1, 1.25, 1.75)),
  min.node.size = c(1, 3, 5, 10), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(.5, .63, .8),
  miss_classification = NA                                               
)

model_list <- list() # Initialize a list to store models

# Execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # Fit model for ith hyperparameter combination
  fit2 <- ranger(
    formula         = is_strike ~ ., 
    data            = df, 
    num.trees       = n_features * 10,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 300344635,
    respect.unordered.factors = 'order',
  )
  # Store model in the list
  model_list[[i]] <- fit2
  
  # Export OOB error 
  hyper_grid$miss_classification[i] <- fit2$prediction.error
}

# Find the model with the lowest misclassification error
best_model_index <- which.min(hyper_grid$miss_classification)
best_model <- model_list[[best_model_index]]

best_model


#### SVM
require(kernlab)
library(e1071)
svm <- caret::train(
  is_strike ~ ., 
  data = df,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 5
)

plot(svm$results$C, svm$results$Accuracy)


svm$bestTune


confusionMatrix(svm)


###### rough work / Possibly need some this below in other models 




dim(train_data)

summary(train_data)
#head(test_data)

pillar::glimpse(train_data)



#pitches <- c("FF", "SL", "CU", "CH", "SI", "FC", "KC", "FS", "EP", "FA", "CS", "KN")
#The numeric equivalents of these vector values
#(as.numeric(factors))
train_data$pitch_type <- train_data$pitch_type %>% 
  factor(c("FF", "SL", "CU", "CH", "SI", "FC", "KC", "FS", "EP", "FA", "CS", "KN"))

# Define the formula for dummy encoding
formula <- ~pitch_type - 1  # '- 1' to avoid an intercept column

# Create a dummyVars object
dummy_model <- dummyVars(formula, data = train_data, fullRank = TRUE)  # fullRank=TRUE ensures n-1 encoding

# Transform the data
dummy_data <- predict(dummy_model, newdata = train_data)

# Convert to dataframe
dummy_df <- as.data.frame(dummy_data)

# If you want to bind the dummy variables back to the original dataframe
train_data <- cbind(dummy_df, train_data)

train_data$pitch.type <- NULL

# Convert is_strike to a factor 
train_data$is_strike <- as.factor(train_data$is_strike)

is.factor(train_data$is_strike)



#### 



## Center and scale data 

preproc <- preProcess(train_data[, -1], method = c("center", "scale"))
transformed_data <- predict(preproc, train_data[, -1])










# need to predict the test_data 

### logistic regression 

require(caret)
require(glmnet)

## Train with 5-Fold CV
set.seed(1)
reguarlized_model <- train(is_strike ~ ., 
                           data = train_data, 
                           method = "glmnet",
                           metric = "Accuracy",
                           tuneLength = 15, ## we'll try 15 alpha and 15 lambda values
                           trControl = trainControl(method = "cv", 
                                                    number = 5,
                                                    search = "random", ## we use random search
                                                    verboseIter = T))

#Aggregating results
#Selecting tuning parameters
#Fitting alpha = 0.206, lambda = 0.0317 on full training set

## Best Tuned Model
reguarlized_model$bestTune

## Train Accuracy
p3 <- predict(reguarlized_model, type = "prob")
p3 <- ifelse(p3[[2]] >= 0.5, T, F)
table(p3, train_data$is_strike)
print(sum(diag(table(p3, train_data$is_strike)))/ nrow(train_data))

####



