#Step 1:Importing the necessary libraries and dataset
library(kernlab)
library(caret)
library(ggplot2)
library(dplyr)
library(caret)
library(cli)
library(e1071)
library(pROC)
library(MASS)
creditcard_data<- read.csv("creditcard.csv")

#Data Exploration
head(creditcard_data)
str(creditcard_data)
summary(creditcard_data)

set.seed(123)
creditcard_data$Class <- as.factor(creditcard_data$Class)
creditcard_data_downsampled <- downSample(x = creditcard_data[, -31], y = creditcard_data$Class)

#Step 2: Data Preprocessing

# The credit card fraud dataset is highly imbalanced, with the majority of transactions being non-
# fraudulent. To balance the dataset, we'll use the `downsample` function from the `caret`
# package to randomly remove observations from the majority class. Additionally, we'll
# standardize the `Amount` and `Time` features to have a mean of 0 and standard deviation of 1.

creditcard_data_downsampled$Amount <- scale(creditcard_data_downsampled$Amount)
creditcard_data_downsampled$Time <- scale(creditcard_data_downsampled$Time)


# Step 3: Splitting the dataset into training and testing sets
# 
# Next, we'll split the preprocessed dataset into training and testing sets, with 80% of the data
# used for training and 20% used for testing.

set.seed(123)
train_index <- createDataPartition(creditcard_data_downsampled$Class, p = 0.8, list = FALSE)
train_set <- creditcard_data_downsampled[train_index, ]
test_set <- creditcard_data_downsampled[-train_index, ]

# Step 4: Building and evaluating models
# Now, we'll build models using the four machine learning algorithms: ANN, SVM, Decision Tree,
# and Logistic Regression. For each model, we'll use cross-validation to tune the hyperparameters
# and choose the best model. Then, we'll evaluate the model on the test set and calculate
# accuracy, precision, recall, and Fl score.


# ANN model

set.seed(123)
ann_model <- train(Class ~ ., data = train_set, method = "nnet", trace = FALSE, 
                   tuneLength = 5, preProcess = c("center", "scale"))
ann_predictions <- predict(ann_model, newdata = test_set)
ann_cm <- confusionMatrix(ann_predictions, test_set$Class)
ann_cm

#SVM model

set.seed(123)
svm_model <- train(Class ~ ., data = train_set, method = "svmRadial", trControl = trainControl(method = "cv"), 
                   preProcess = c("center", "scale"))
svm_predictions <- predict(svm_model, newdata = test_set)
svm_cm <- confusionMatrix(svm_predictions, test_set$Class)
svm_cm


#Decision Tree model specificities

set.seed(123)
dt_model <- train(Class ~ ., data = train_set, method = "rpart", trControl = trainControl(method = "cv"))
dt_predictions <- predict(dt_model, newdata = test_set)
dt_cm <- confusionMatrix(dt_predictions, test_set$Class)
dt_cm



 
### Logistic Regression model

set.seed(123)
lr_model <- train(Class ~ ., data = train_set, method = "glm", family = "binomial", 
                  trControl = trainControl(method = "cv"), preProcess = c("center", "scale"))
lr_predictions <- predict(lr_model, newdata = test_set)
lr_cm <- confusionMatrix(lr_predictions, test_set$Class)
lr_cm






# we're using the `train` function from the `caret` package to train a logistic
# regression model on the preprocessed training set. We're setting the `method` parameter to
# "glm" to indicate that we want to use a generalized linear model with the binomial family for
# logistic regression. We're also using cross-validation to tune the hyperparameters and choose
# the best model, and standardizing the features using the `preprocess` parameter.
# Then, we're using the trained model to make predictions on the test set and storing the
# predictions in`lr_predicitons` Finally, we're using the `confusionnatrix` function from the
# `caret` package to calculate the confusion matrix for the logistic regression model and storing
# the result in `lr_cm`.

# Accuracy, recall, precision, and F1 score for ANN
ann_acc <- ann_cm$overall['Accuracy']
ann_sens <- ann_cm$byClass['Sensitivity']
ann_spec <- ann_cm$byClass['Specificity']
ann_prec <- ann_cm$byClass['Pos Pred Value']
ann_f1 <- 2 * (ann_prec * ann_sens) / (ann_prec + ann_sens)

lr_acc*100
lr_sens*100
lr_spec*100
lr_prec*100
lr_f1*100

library(ggplot2)

ann_metrics <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"),
  Value = c(ann_acc, ann_sens, ann_spec, ann_prec, ann_f1)
)

ann_metrics






# Accuracy, recall, precision, and F1 score for SVM
svm_acc <- svm_cm$overall['Accuracy']
svm_sens <- svm_cm$byClass['Sensitivity']
svm_spec <- svm_cm$byClass['Specificity']
svm_prec <- svm_cm$byClass['Pos Pred Value']
svm_f1 <- 2 * (svm_prec * svm_sens) / (svm_prec + svm_sens)

# Accuracy, recall, precision, and F1 score for decision tree
dt_acc <- dt_cm$overall['Accuracy']
dt_sens <- dt_cm$byClass['Sensitivity']
dt_spec <- dt_cm$byClass['Specificity']
dt_prec <- dt_cm$byClass['Pos Pred Value']
dt_f1 <- 2 * (dt_prec * dt_sens) / (dt_prec + dt_sens)

# Accuracy, recall, precision, and F1 score for logistic regression
lr_acc <- lr_cm$overall['Accuracy']
lr_sens <- lr_cm$byClass['Sensitivity']
lr_spec <- lr_cm$byClass['Specificity']
lr_prec <- lr_cm$byClass['Pos Pred Value']
lr_f1 <- 2 * (lr_prec * lr_sens) / (lr_prec + lr_sens)

# Here, we're using the function from the caret' package to obtain
# the confusion matrices for each of the models. We're then extracting the accuracy, recall
# (sensitivity), precision (positive predictive value), and Fl score for each model from the
# confusion matrices.
# To compare the performance of the models, you can create a table that shows the accuracy,
# recall, precision, and Fl score for each model side by side:
# 
# # Create a table to compare the performance of the models


comparison_table <- data.frame(Model = c("ANN", "SVM", "Decision Tree", "Logistic Regression"),
                               Accuracy = c(ann_acc, svm_acc, dt_acc, lr_acc),
                               Recall = c(ann_sens, svm_sens, dt_sens, lr_sens),
                               Precision = c(ann_prec, svm_prec, dt_prec, lr_prec),
                               F1_Score = c(ann_f1, svm_f1, dt_f1, lr_f1))

print(comparison_table)

# Convert the data frame to "long" format
comparison_table_long <- tidyr::gather(comparison_table, metric, value, -Model)

# Create the graph
ggplot(data = comparison_table_long, aes(x = Model, y = value, fill = metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Model", y = "Metric value", fill = "") +
  scale_fill_manual(values = c("#F8766D", "#00BFC4", "#619CFF", "#7CAE00")) +
  theme_classic()






