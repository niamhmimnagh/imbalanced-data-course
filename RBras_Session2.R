############################################################
# 1. SETUP & DATA IMPORT
############################################################

# Load required libraries
library(caret)        # For ML workflows
library(dplyr)        # For data manipulation
library(nestedcv)     # For SMOTE
library(ggplot2)      # For plotting
library(randomForest) # For Random Forest model
library(e1071)        # Required by caret
library(xgboost)      # For gradient boosting
library(pROC)         # For ROC curves
library(PRROC)        # For Precision-Recall curves
library(isotree)      # Isolation Forest for outlier detection
library(DescTools)    # Misc tools (used by LOF)
library(abodOutlier)  # Angle-Based Outlier Detection
library(glmnet)       # For elastic net model
library(keras)        # For autoencoder (deep learning)

# Set seed for reproducibility
set.seed(123)

# Load GermanCredit dataset (imbalanced binary classification)
data(GermanCredit)
GermanCredit$Class <- factor(GermanCredit$Class, levels = c("Bad", "Good"))

# Inspect structure and class distribution
str(GermanCredit)
table(GermanCredit$Class)


############################################################
# 2. EXPLORING CLASS IMBALANCE
############################################################

# Check class proportions
prop.table(table(GermanCredit$Class))

# Bar plot of class imbalance
ggplot(GermanCredit, aes(x = Class)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() 


############################################################
# 3. RESAMPLING TO FIX CLASS IMBALANCE
############################################################

# Separate predictors and response
X <- GermanCredit %>% dplyr::select(-Class)
y <- GermanCredit$Class

# Downsampling
down_train <- downSample(x = X, y = y)

# Upsampling
up_train <- upSample(x = X, y = y)

# SMOTE (Synthetic Minority Over-sampling Technique)
X_mat <- model.matrix(Class ~ . - 1, data = GermanCredit)  # Convert factors to dummy variables
smote_train <- smote(y = y, x = X_mat)
smote_train <- cbind.data.frame("Class" = smote_train$y, smote_train$x)

# Visualise results of resampling
par(mfrow = c(1, 3))
barplot(table(down_train$Class), main = "Downsampled", col = "tomato", ylim = c(0, 700))
barplot(table(up_train$Class), main = "Upsampled", col = "lightgreen")
barplot(table(smote_train$Class), main = "SMOTE", col = "skyblue")
par(mfrow = c(1, 1))


############################################################
# 4. TRAIN/TEST SPLIT
############################################################

# 70% train / 30% test split on SMOTE-balanced data
set.seed(123)
train_idx <- createDataPartition(smote_train$Class, p = 0.7, list = FALSE)
train_data <- smote_train[train_idx, ]
test_data  <- GermanCredit[-train_idx, ]  # Use original imbalanced data for testing

# Also create imbalanced training set for comparison
train_idx_imbalanced <- createDataPartition(GermanCredit$Class, p = 0.7, list = FALSE)
train_data_imbalanced <- GermanCredit[train_idx_imbalanced, ]


############################################################
# 5. MODEL EVALUATION FUNCTION
############################################################

evaluate_model <- function(pred_probs, true_labels, threshold = 0.5) {
  # Convert probabilities to predicted classes
  pred_class <- ifelse(pred_probs > threshold, "Bad", "Good")
  
  # Confusion matrix
  cm <- confusionMatrix(factor(pred_class, levels = c("Bad", "Good")), true_labels, positive = 'Bad')
  print(cm)
  
  # F1, Precision, Recall
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  cat("\nPrecision:", round(precision, 3),
      "\nRecall:", round(recall, 3),
      "\nF1-score:", round(f1, 3), "\n")
  
  # ROC Curve
  roc_obj <- roc(true_labels, pred_probs, levels = c("Bad", "Good"), direction = ">")
  plot(roc_obj, main = "ROC Curve")
  
  # Precision-Recall Curve
  pr <- pr.curve(scores.class0 = pred_probs[true_labels == "Bad"],
                 scores.class1 = pred_probs[true_labels == "Good"],
                 curve = TRUE)
  plot(pr)
}


############################################################
# 6. RANDOM FOREST MODELS
############################################################

# Train on SMOTE-balanced data
rf_model <- randomForest(Class ~ ., data = train_data, ntree = 1000)
rf_probs <- predict(rf_model, test_data %>% dplyr::select(-Class), type = "prob")[, "Bad"]
evaluate_model(rf_probs, test_data$Class)

# Train on original imbalanced data
rf_model <- randomForest(Class ~ ., data = train_data_imbalanced, ntree = 1000)
rf_probs <- predict(rf_model, test_data %>% dplyr::select(-Class), type = "prob")[, "Bad"]
evaluate_model(rf_probs, test_data$Class)


############################################################
# 7. OUTLIER DETECTION ON ORIGINAL DATA
############################################################

# Use only numeric predictors
X_num <- GermanCredit %>% dplyr::select_if(is.numeric)
y_true <- GermanCredit$Class  
y_true_bin <- as.factor(ifelse(y_true == "Bad", 1, 0)) 

# Isolation Forest
iso_model <- isolation.forest(X_num, nthreads = 1)
iso_scores <- predict(iso_model, X_num)
iso_outliers <- iso_scores > quantile(iso_scores, 0.7)
iso_pred_bin <- as.factor(as.numeric(iso_outliers))
confusionMatrix(iso_pred_bin, y_true_bin, positive = "1")

# Local Outlier Factor (LOF) - tune k
k_vals <- seq(10, 150, by = 10)
results <- data.frame(k = k_vals, F1 = NA)

for (i in seq_along(k_vals)) {
  k <- k_vals[i]
  scores <- LOF(X_num, k = k)
  pred_bin <- as.factor(as.numeric(scores >= quantile(scores, 0.7)))
  cm <- confusionMatrix(pred_bin, y_true_bin, positive = "1")
  results$F1[i] <- cm$byClass["F1"]
}

# Plot F1 vs k
plot(results$k, results$F1, type = "b", main = "F1 vs k in LOF", xlab = "k", ylab = "F1")

# Final LOF model
lof_scores <- LOF(X_num, k = 150)
lof_outliers <- lof_scores >= quantile(lof_scores, 0.7)
lof_pred_bin <- as.factor(as.numeric(lof_outliers))
confusionMatrix(lof_pred_bin, y_true_bin, positive = "1")

# ABOD (Angle-Based Outlier Detection)
abod_scores <- abod(as.matrix(X_num))
abod_outliers <- abod_scores < quantile(abod_scores, 0.05)  # lower scores = more likely outlier


############################################################
# 9. STACKING ENSEMBLE
############################################################

# 9.1 Random Forest
rf_model <- randomForest(Class ~ ., data = train_data, ntree = 100)
rf_probs <- predict(rf_model, test_data %>% dplyr::select(-Class), type = "prob")[, "Bad"]

# 9.2 Elastic Net
x_train <- model.matrix(Class ~ . - 1, data = train_data)
y_train <- ifelse(train_data$Class == "Bad", 1, 0)
x_test  <- model.matrix(Class ~ . - 1, data = test_data)
cv_glmnet <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5)
enet_probs <- predict(cv_glmnet, newx = x_test, s = "lambda.min", type = "response")[, 1]

# 9.3 XGBoost
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test)
xgb_model <- xgboost(data = dtrain, nrounds = 50, objective = "binary:logistic", verbose = 0)
xgb_probs <- predict(xgb_model, dtest)

# 9.4 Meta-learner: Logistic Regression on model outputs
stack_input <- data.frame(rf = rf_probs, enet = enet_probs, xgb = xgb_probs)
stack_y <- ifelse(test_data$Class == "Bad", 1, 0)
stack_model <- glm(stack_y ~ ., data = stack_input, family = "binomial")
stack_probs <- predict(stack_model, newdata = stack_input, type = "response")

# 9.5 Evaluate all models
cat("\n=== Random Forest ===\n")
evaluate_model(rf_probs, test_data$Class)

cat("\n=== Elastic Net ===\n")
evaluate_model(enet_probs, test_data$Class)

cat("\n=== XGBoost ===\n")
evaluate_model(xgb_probs, test_data$Class)

cat("\n=== Stacked Model ===\n")
evaluate_model(stack_probs, test_data$Class)


############################################################
# 10. AUTOENCODER FOR ANOMALY DETECTION (NOT RUN)
############################################################

# Scale numeric data
X_scaled <- scale(X_num)

# Train/test split
set.seed(123)
idx <- sample(1:nrow(X_scaled), sise = floor(0.8 * nrow(X_scaled)))
X_train <- X_scaled[idx, ]
X_test  <- X_scaled[-idx, ]

# Autoencoder architecture
input_dim <- ncol(X_train)
encoding_dim <- round(input_dim / 2)

input_layer <- layer_input(shape = input_dim)
encoder <- input_layer %>%
  layer_dense(units = encoding_dim, activation = "relu") %>%
  layer_dense(units = round(encoding_dim / 2), activation = "relu")

decoder <- encoder %>%
  layer_dense(units = encoding_dim, activation = "relu") %>%
  layer_dense(units = input_dim, activation = "linear")

autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)
autoencoder_model %>% compile(loss = "mean_squared_error", optimiser = "adam")

# Train model
history <- autoencoder_model %>% fit(
  x = X_train, y = X_train,
  epochs = 50, batch_sise = 32,
  validation_split = 0.2, verbose = 0
)

# Calculate reconstruction error on test set
X_test_pred <- autoencoder_model %>% predict(X_test)
recon_error <- rowMeans((X_test - X_test_pred)^2)

# Set threshold from training error
X_train_pred <- autoencoder_model %>% predict(X_train)
train_error <- rowMeans((X_train - X_train_pred)^2)
threshold <- quantile(train_error, 0.95)

# Flag outliers
auto_outliers <- recon_error > threshold
cat("Autoencoder outliers detected:", sum(auto_outliers), "\n")

# Plot reconstruction error
hist(recon_error, breaks = 50,
     main = "Reconstruction Errors (Test Set)",
     xlab = "Reconstruction Error", col = "grey")
abline(v = threshold, col = "red", lwd = 2)
