library(rio)
library(dplyr)
library(MASS)
library(car)
library(QuantPsyc)
library(sandwich)
library(glmnet)
library(e1071)
library(rpart)
library(kernlab)
library(tidyverse)
library(caret)
library(xgboost)
library(class)
library(caret)
library(olsrr)
library(qpcR)
library(mosaic)
library(mosaicModel)
library(ISLR)
library(bestglm)
library(glmulti)
library(rminer)


data1 <- import("~/Desktop/Fall 2020 Recovery and RPE Scores - F19.csv")

#residuals
#t-test
#anova

allDataNoLag <- data1[c(10:28)] #all data for same day (No REC and No Lag)
allDataNoLag[] <- lapply(allDataNoLag[], function(x) round(as.numeric(x),2)) #transform '-' into NA's

allDataNoLag <- allDataNoLag %>% dplyr::rename(Distance = `Distance Total`) #2407x19

allDataNoNAs <- na.omit(allDataNoLag) #1716x19

########################################Removing Outliers

#Removing avg hr, max hr, heart rate exertion
testAllData <- allDataNoNAs[-c(11,12,15)] #1716x16


boxplot(testAllData$RPE)
boxplot(testAllData$`Dynamic Stress Load`)
for (i in names(testAllData[c(2:16)])) {
  #print(testAllData[,1])
  Q <- quantile(testAllData[,i], probs=c(.25, .75), na.rm = FALSE)
  iqr <- IQR(testAllData[,i])
  eliminated<- subset(testAllData, testAllData[,i] > (Q[1] - 1.5*iqr) 
                      & testAllData[i] < (Q[2]+1.5*iqr))
  boxplot(eliminated[,i])
}
#eliminated = 1589x16 

lmElim <- lm(RPE ~ Distance + `Max Speed` + `HML Distance` + `HML Efforts` + `Sprint Distance` + Sprints + 
               Accelerations + Decelerations +  `Average Metabolic Power` + `Dynamic Stress Load` + 
               `High Speed Running (Relative)` + `HML Density` + `Speed Intensity` + 
               Impacts + Duration, data = eliminated)


outlierTest(lmElim) #but still has bad max speeds

cutoff <- 4/((nrow(eliminated)-length(lmElim$coefficients)-1))
plot(lmElim, which = 4, cook.levels = cutoff)

########################################Normality and Plotting
###External
plot(rstandard(lmElim))
par(mfrow=c(1,1))
qqnorm(rstudent(lmElim), ylab="Standardized Residuals") #Plot normal prob with external residuals
qqline(rstudent(lmElim))

#External residuals vs fitted for all 19
plot(rstudent(lmElim) ~ fitted(lmElim), xlab = "Fitted Values", ylab = "Studentized Residuals", main = "Residuals vs Fitted (19)")

#External vs each variable
par(mfrow=c(3,3))
plot(rstudent(lmElim) ~ eliminated$RPE, ylab="Studentized Residuals", xlab = "RPE", main = "Residuals vs RPE")
plot(rstudent(lmElim) ~ eliminated$Duration, ylab="Studentized Residuals", xlab = "Duration", main = "Residuals vs Duration")
plot(rstudent(lmElim) ~ eliminated$Distance, ylab="Studentized Residuals", xlab = "Distance", main = "Residuals vs Distance")
plot(rstudent(lmElim) ~ eliminated$`Max Speed`, ylab="Studentized Residuals", xlab = "Max Speed", main = "Residuals vs Max Speed")
plot(rstudent(lmElim) ~ eliminated$`HML Distance`, ylab="Studentized Residuals", xlab = "HML Distance", main = "Residuals vs HML Distance")
plot(rstudent(lmElim) ~ eliminated$`HML Efforts`, ylab="Studentized Residuals", xlab = "HML Efforts", main = "Residuals vs HML Efforts")
plot(rstudent(lmElim) ~ eliminated$`Sprint Distance`, ylab="Studentized Residuals", xlab = "Sprint Distance", main = "Residuals vs Sprint Distance")
plot(rstudent(lmElim) ~ eliminated$Sprints, ylab="Studentized Residuals", xlab = "Sprints", main = "Residuals vs Sprints")
plot(rstudent(lmElim) ~ eliminated$Accelerations, ylab="Studentized Residuals", xlab = "Accelerations", main = "Residuals vs Accelerations")
plot(rstudent(lmElim) ~ eliminated$Decelerations, ylab="Studentized Residuals", xlab = "Decelerations", main = "Residuals vs Decelerations")
plot(rstudent(lmElim) ~ eliminated$`Average Metabolic Power`, ylab="Studentized Residuals", xlab = "Average Metabolic Power", main = "Residuals vs Average Metabolic Power")
plot(rstudent(lmElim) ~ eliminated$`Dynamic Stress Load`, ylab="Studentized Residuals", xlab = "Dynamic Stress Load", main = "Residuals vs Dynamic Stress Load")
plot(rstudent(lmElim) ~ eliminated$`High Speed Running (Relative)`, ylab="Studentized Residuals", xlab = "High Speed Running (Relative)", main = "Residuals vs High Speed Running (Relative)")
plot(rstudent(lmElim) ~ eliminated$`HML Density`, ylab="Studentized Residuals", xlab = "HML Density", main = "Residuals vs HML Density")
plot(rstudent(lmElim) ~ eliminated$`Speed Intensity`, ylab="Studentized Residuals", xlab = "Speed Intensity", main = "Residuals vs Speed Intensity")
plot(rstudent(lmElim) ~ eliminated$Impacts, ylab="Studentized Residuals", xlab = "Impacts", main = "Residuals vs Impacts")

########################################Splitting into 3 datasets from model validation

set.seed(42)

splits = c(train = .6, test = .2, validate = .2)

#random assigning
?sample
g = sample(cut(
  seq(nrow(eliminated)), 
  nrow(eliminated)*cumsum(c(0,splits)),
  labels = names(splits)
))

res = split(eliminated, g)

sapply(res, nrow)/nrow(eliminated)
addmargins(prop.table(table(g)))

res$train #953x16
res$test #318x16
res$validate #318x16

lmTrain <- lm(RPE ~ Distance + `Max Speed` + `HML Distance` + `HML Efforts` + `Sprint Distance` + Sprints + 
                Accelerations + Decelerations +  `Average Metabolic Power` + `Dynamic Stress Load` + 
                `High Speed Running (Relative)` + `HML Density` + `Speed Intensity` + 
                Impacts + Duration, data = res$train)

lmTest <- lm(RPE ~ Distance + `Max Speed` + `HML Distance` + `HML Efforts` + `Sprint Distance` + Sprints + 
               Accelerations + Decelerations +  `Average Metabolic Power` + `Dynamic Stress Load` + 
               `High Speed Running (Relative)` + `HML Density` + `Speed Intensity` + 
               Impacts + Duration, data = res$test)

########################################All Possible Regressions

##All possible regressions
allReg <- ols_step_all_possible(lmTrain)
allRegTrain <- allReg
View(allRegTrain)
plot(allRegTrain)
allRegTrain$model
allRegTrain[allRegTrain$mindex == 32192]

min(allRegTrain$aic[allRegTrain$n == 15])
s1 <- allRegTrain[allRegTrain$n == 14]

# Sort by column index [2] then [5]
sortedART <- allRegTrain[order( allRegTrain[,5], allRegTrain[,2] ),]

k11 <- lm(RPE ~ Distance + `HML Distance` + `HML Efforts` +
            Accelerations + Decelerations +  `Average Metabolic Power` + `Dynamic Stress Load` + 
            `HML Density` + `Speed Intensity` + Impacts + Duration, data = res$train)
k12 <- lm(RPE ~ Distance + `HML Distance` + `HML Efforts` + Sprints + 
            Accelerations + Decelerations +  `Average Metabolic Power` + `Dynamic Stress Load` + 
            `HML Density` + `Speed Intensity` + Impacts + Duration, data = res$train)
PRESS1(k11) #press = 2047.351
PRESS1(k12) #press = 2048.812
########################################Forward AIC

forwardTrain <- ols_step_forward_aic(lmTrain)
forwardTrain
sort(forwardTrain$model$coefficients)
########################################Backward AIC

backwardTrain <- ols_step_backward_aic(lmTrain)
backwardTrain
sort(backwardTrain$model$coefficients)
########################################Stepwise AIC

stepwiseTrain0 <- lm(RPE ~ 1, data = res$train) #empty modfit
stepwiseTrainAll <- lm(RPE ~ ., data = res$train) #full modfit , data = stddata?
#stddata <- scale(res$train)
#lm.beta(stepwiseTrainAll)

stepwiseTrainBest <- stepAIC(stepwiseTrainAll, direction="both",
                             scope=list(upper=stepwiseTrainAll, lower=stepwiseTrain0) ) #finds best modfit
stepwiseTrainBest
formula(stepwiseTrainBest)
summary(stepwiseTrainBest)
sort(coefficients(stepwiseTrainBest))

Rsq.ad(stepwiseTrainBest)
AIC(stepwiseTrainBest)

########################################Normal GLM

summary(glmTrain <- glm(RPE ~ Distance + `Max Speed` + `HML Distance` + `HML Efforts` + `Sprint Distance` + Sprints + 
                          Accelerations + Decelerations +  `Average Metabolic Power` + `Dynamic Stress Load` + 
                          `High Speed Running (Relative)` + `HML Density` + `Speed Intensity` + 
                          Impacts + Duration, data = res$train))
glmTrain$formula
sort(glmTrain$coefficients) #removed max speed, sprint distance, sprints, high speed running, 
Rsq.ad(glmTrain)
AIC(glmTrain)

#bestglm::bestglm(res$train[,-1], IC = "AIC")

#Add the deviance residuals versus each regressor
#and normal probability plot of the residuals.

########################################Poisson Regression

summary(poissonTrain <- glm(RPE ~ Distance + `Max Speed` + `HML Distance` + `HML Efforts` + `Sprint Distance` + Sprints + 
                              Accelerations + Decelerations +  `Average Metabolic Power` + `Dynamic Stress Load` + 
                              `High Speed Running (Relative)` + `HML Density` + `Speed Intensity` + 
                              Impacts + Duration, family = "poisson", data = res$train))

sort(poissonTrain$coefficients) #removed max speed, sprint distance, sprints, high speed running, HMLDe, impacts

Rsq.ad(poissonTrain)
AIC(poissonTrain)

?optim
?Multinomial

########################################Penalized Regression

xTrain <- model.matrix(RPE~., res$train)[,-1]
yTrain <- res$train$RPE
glmnet(xTrain, yTrain, alpha = 1, lambda = NULL)

########################################Ridge

cvTrain <- cv.glmnet(xTrain, yTrain, alpha = 0)
# Display the best lambda value
cvTrain$lambda.min

# Fit the final model on the training data
ridgeTrain <- glmnet(xTrain, yTrain, alpha = 0, lambda = cvTrain$lambda.min)
# Display regression coefficients
coef(ridgeTrain)
ridgeTrain

tLLTrain <- ridgeTrain$nulldev - deviance(ridgeTrain)
kTrain <- ridgeTrain$df
nTrain <- ridgeTrain$nobs
AICcTrain <- -tLLTrain+2*kTrain+2*kTrain*(kTrain+1)/(nTrain-kTrain-1)
AICcTrain

Rsq.ad(ridgeTrain)
rsquared(ridgeTrain)
adjr2 <- ridgeTrain$dev.ratio[which(ridgeTrain$lambda == ridgeTrain$lambda)]
adjr2

# Make predictions on the test data
x.test <- model.matrix(RPE ~., test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$RPE),
  Rsquare = R2(predictions, test.data$RPE)
)

########################################Lasso

cvTrain <- cv.glmnet(xTrain, yTrain, alpha = 1)
# Display the best lambda value
cvTrain$lambda.min

# Fit the final model on the training data
lassoTrain <- glmnet(xTrain, yTrain, alpha = 1, lambda = cvTrain$lambda.min)
# Dsiplay regression coefficients
coef(lassoTrain)

tLLTrainLasso <- lassoTrain$nulldev - deviance(lassoTrain)
kTrainLasso <- lassoTrain$df
nTrainLasso <- lassoTrain$nobs
AICcTrainLasso <- -tLLTrainLasso+2*kTrainLasso+2*kTrainLasso*(kTrainLasso+1)/(nTrainLasso-kTrainLasso-1)
AICcTrainLasso

adjr2lasso <- lassoTrain$dev.ratio[which(lassoTrain$lambda == lassoTrain$lambda)]
adjr2lasso



# Make predictions on the test data
x.test <- model.matrix(RPE ~., test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$RPE),
  Rsquare = R2(predictions, test.data$RPE)
)

########################################Elastic Net

# Build the model using the training set
set.seed(42)
elasticTrain <- train(
  RPE ~., data = res$train, method = "glmnet",
  trControl = trainControl("cv", number = 5),
  tuneLength = 10
)
# Best tuning parameter
elasticTrain$bestTune
elasticTrain

# Coefficient of the final model. You need
# to specify the best lambda
coef(elasticTrain$finalModel, elasticTrain$bestTune$lambda)

tLLTrainElastic <- elasticTrain$finalModel$nulldev - deviance(elasticTrain$finalModel)
deviance(elasticTrain$finalModel)
kTrainElastic <- elasticTrain$finalModel$df
nTrainElastic <- elasticTrain$finalModel$nobs
AICcTrainElastic <- -tLLTrainElastic+2*kTrainElastic+2*kTrainElastic*(kTrainElastic+1)/(nTrainElastic-kTrainElastic-1)
AICcTrainElastic


get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

get_best_result(elasticTrain)

# Make predictions on the test data
x.test <- model.matrix(RPE ~., test.data)[,-1]
predictions <- model %>% predict(x.test)
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$RPE),
  Rsquare = R2(predictions, test.data$RPE)
)

########################################Using Ridge, Lasso, and Elastic Net in One Computation

#Computing all 3
lambda <- 10^seq(-3, 3, length = 100)

# Build the model
set.seed(42)
ridge <- train(
  RPE ~., data = train.data, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda)
)
# Model coefficients
coef(ridge$finalModel, ridge$bestTune$lambda)
# Make predictions
predictions <- ridge %>% predict(test.data)
# Model prediction performance
data.frame(
  RMSE = RMSE(predictions, test.data$RPE),
  Rsquare = R2(predictions, test.data$RPE)
)

# Build the model
set.seed(42)
lasso <- train(
  RPE ~., data = train.data, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda)
)
# Model coefficients
coef(lasso$finalModel, lasso$bestTune$lambda)
# Make predictions
predictions <- lasso %>% predict(test.data)
# Model prediction performance
data.frame(
  RMSE = RMSE(predictions, test.data$RPE),
  Rsquare = R2(predictions, test.data$RPE)
)

# Build the model
set.seed(42)
elastic <- train(
  RPE ~., data = train.data, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Model coefficients
coef(elastic$finalModel, elastic$bestTune$lambda)
# Make predictions
predictions <- elastic %>% predict(test.data)
# Model prediction performance
data.frame(
  RMSE = RMSE(predictions, test.data$RPE),
  Rsquare = R2(predictions, test.data$RPE)
)

#Comparing models
models <- list(ridge = ridge, lasso = lasso, elastic = elastic)
resamples(models) %>% summary( metric = "RMSE")

#Lasso is best model since it has the lowest median RMSE

########################################SVM

svm.model <- svm(RPE ~ ., data = res$train, cost = 100, gamma = 1)
svm.model
plot(svm.model, res$train)

svmTrain <- fit(RPE ~ ., data = res$train, model = "svm", kpar=list(sigma=0.10), C=2)
svmImp <- Importance(svmTrain, data = res$train)
svmImp$interactions
print(round(svmImp$imp,digits = 2))
tune.svm(svm.model)
tune(svm.model)
W <- t(svm.model$coefs) %*% svm.model$SV
sort(W)

svm.model

b <- svm.model$rho
b


OptModelsvm <- tune(svm, RPE~., data=res$train,ranges=list(elsilon=seq(0,1,0.1), cost=1:100))
OptModelsvm
bestSVM <- OptModelsvm$best.model
bestSVM

Rsq.ad(svm.model)
2*RMSE(svm.model)^2 / RMSE(svm.model)
RMSE(bestSVM)
aicSVM <- ((2*10) - 16*log(RMSE(svm.model)))*16
aicSVM

training <- res$train
training[["RPE"]] <- factor(training[["RPE"]])
summary(training)

set.seed(42)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

svm_Linear <- train(RPE ~., data = res$train, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_Linear

test_predSVM <- predict(svm_Linear, newdata = res$test)

u <- union(test_predSVM, res$test$RPE)
t <- table(factor(test_predSVM, u), factor(res$test$RPE, u))
confusionMatrix(t)

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Linear_Grid <- train(RPE ~., data = res$train, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_Linear_Grid #0.01 is best
plot(svm_Linear_Grid)

test_pred_grid <- predict(svm_Linear_Grid, newdata = res$test)
test_pred_grid

u1 <- union(test_pred_grid, res$test$RPE)
t1 <- table(factor(test_pred_grid, u1), factor(res$test$RPE, u1))
confusionMatrix(t1)

########################################Boosted Forests

#Fitting the model on training set
set.seed(42)
boostedTrain <- train(
  RPE ~., data = res$train, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
# Best tuning parameter
boostedTrain$bestTune
summary(boostedTrain$results)
boostedTrain$results
boostedTrain$bestTune
boostedTrain$finalModel

aicBoosted <- ((2*8) - 16*log(RMSE(boostedTrain)))*16
aicBoosted
#Variable importance
varImp(boostedTrain$bestTune)
varImp(boostedTrain)
varImp(boostedTrain$finalModel)
xgb.importance(names(res$train[,2:16]), boostedTrain$finalModel)
?xgb.importance
#shapeley score/value




RMSE(boostedTrain)
# Make predictions on the test data
predictions <- boostedTrain %>% predict(res$test)
head(predictions)
# Compute model prediction accuracy rate
cor(predicted, res$test$RPE)
# Compute the average prediction error RMSE
RMSE(predictions, test.data$RPE)
#compare
RMSE(predictions, res$train$RPE) #test is not higher than train, no evidence of overfit


######################################## KNN

#scale
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

train.loan <- res$train # 70% training data
test.loan <- res$test # remaining 30% test data

#Creating seperate dataframe for rpe feature which is the target.
train.loan_labels <- res$train[,1]
train.loan_labels
test.loan_labels <-res$test[,1]
test.loan_labels

#Find the number of observation
NROW(train.loan_labels) #30.85, rows 952
?knn
knn.30 <- knn(train=train.loan, test=test.loan, cl=train.loan_labels, k=30)
knn.31 <- knn(train=train.loan, test=test.loan, cl=train.loan_labels, k=31)

#Calculate the proportion of correct classification for k = 30, 31
ACC.30 <- 100 * sum(test.loan_labels == knn.30)/NROW(test.loan_labels)
ACC.31 <- 100 * sum(test.loan_labels == knn.31)/NROW(test.loan_labels)

ACC.30
ACC.31


# Check prediction against actual value in tabular form for k=30
table(knn.30 ,test.loan_labels)
cor(as.numeric(knn.30), test.loan_labels)

# Check prediction against actual value in tabular form for k=31
table(knn.31 ,test.loan_labels)
cor(as.numeric(knn.31), test.loan_labels)


confusionMatrix(table(knn.31 ,test.loan_labels))
uKNN <- union(knn.31, res$test$RPE)
tKNN <- table(factor(knn.31, uKNN), factor(res$test$RPE, uKNN))
confusionMatrix(tKNN)


#optimizing the accuracy
i=1
k.optm=1
set.seed(42)
for (i in 1:75){
  knn.mod <- knn(train=train.loan, test=res$test, cl=train.loan_labels, k=i)
  k.optm[i] <- 100 * sum(test.loan_labels == knn.mod)/NROW(test.loan_labels)
  k=i
  cat(k,'=',k.optm[i],'
      ')
}
#best is k = 44, acc = 31.76101

knn.44 <- knn(train=train.loan, test=test.loan, cl=train.loan_labels, k=44)
ACC.44 <- 100 * sum(test.loan_labels == knn.44)/NROW(test.loan_labels)
ACC.44
uKNN44 <- union(knn.44, res$test$RPE)
tKNN44 <- table(factor(knn.44, uKNN44), factor(res$test$RPE, uKNN44))
confusionMatrix(tKNN44)
########################################Test Datasets From Top K of Previous Approaches

########All Possible Regression
allTestLm <- lm(RPE ~ Distance + `HML Distance` + `HML Efforts` + Sprints +
                  Accelerations + Decelerations + `Average Metabolic Power` + `Dynamic Stress Load` + 
                  `HML Density` + `Speed Intensity` + Impacts + Duration, data = res$test)
allRegTest <- ols_step_all_possible(allTestLm)
allRegTest
RMSE(allTestLm) #1.258752
count <- summary(influence.measures(allTestLm)) #33
outlierTest(allTestLm)
########Forward AIC
forwardTestlm <- lm(RPE ~ `Average Metabolic Power` + Decelerations + `HML Density` +
                      `Dynamic Stress Load` + Impacts + `Max Speed` + Accelerations +
                      `HML Efforts` + Distance + Duration + `Speed Intensity` + `HML Distance`, data = res$test)
forwardTest <- ols_step_forward_aic(forwardTestlm) #RPE ~ , data = res$test
forwardTest
RMSE(forwardTestlm) #1.274529
count1 <- summary(influence.measures(forwardTestlm)) #32
outlierTest(forwardTestlm)
########Backward AIC
backwardTestLm <- lm(RPE ~ `Average Metabolic Power`  + `HML Density` + Decelerations + Accelerations +
                       `Dynamic Stress Load` + Distance + `HML Distance` + Duration + Impacts +
                       `HML Efforts` + `Speed Intensity`, data = res$test)
backwardTest <- ols_step_backward_aic(lmTest) #RPE ~ , data = res$test
backwardTest
RMSE(backwardTestLm) #1.277867
count10 <- summary(influence.measures(backwardTestLm)) #32
outlierTest(backwardTestLm)
########Stepwise AIC
stepwiseTest0 <- lm(RPE ~ 1, data = res$test) #empty modfit
stepwiseTestAll <- lm(RPE ~ `Average Metabolic Power`  + `HML Density` + Decelerations + Accelerations +
                        `Dynamic Stress Load` + Distance + `HML Distance` + Duration + Impacts +
                        `HML Efforts` + `Speed Intensity`, data = res$test) #full modfit , add the variables here so it isn't all
stepwiseTestBest <- stepAIC(stepwiseTestAll, direction="both",
                            scope=list(upper=stepwiseTestAll, lower=stepwiseTest0) ) #finds best modfit
formula(stepwiseTestBest)
summary(stepwiseTestBest)
coefficients(stepwiseTestBest)

RMSE(stepwiseTestBest) #1.280747
count2 <- summary(influence.measures(stepwiseTestBest)) #41
outlierTest(stepwiseTestBest)
########Normal GLM
summary(glmTest <- glm(RPE ~ `Average Metabolic Power`  + `HML Density` + Decelerations + Accelerations +
                         `Dynamic Stress Load` + Distance + `HML Distance` + Duration + Impacts +
                         `HML Efforts` + `Speed Intensity`, data = res$test))
RMSE(glmTest) #1.277867
count3 <- summary(influence.measures(glmTest)) #32
outlierTest(glmTest)
########Poisson Regression
summary(poissonTest <- glm(RPE ~ `Average Metabolic Power` + Decelerations + Accelerations +
                             `Dynamic Stress Load` + Distance + `HML Distance` + Duration +
                             `HML Efforts` + `Speed Intensity` , family = "poisson", data = res$test))
RMSE(poissonTest) #0.6563605
count4 <- summary(influence.measures(poissonTest)) #40
outlierTest(poissonTest)
########Penalized Regression
################Ridge
xTest <- model.matrix(RPE~., res$test)[,-1]
yTest <- res$test$RPE
glmnet(xTest, yTest, alpha = 1, lambda = NULL)
cvTest <- cv.glmnet(xTest, yTest, alpha = 0)
# Display the best lambda value
cvTest$lambda.min
# Fit the final model on the test data
ridgeTest <- glmnet(xTest, yTest, alpha = 0, lambda = cvTest$lambda.min)
# Display regression coefficients
coef(ridgeTest)


lambda <- 10^seq(-3, 3, length = 100)
# Build the model
set.seed(42)
ridgeTest1 <- train(
  RPE ~., data = res$test, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda)
)


RMSE(ridgeTest1) #1.262414
count5 <- summary(influence.measures(ridgeTest1$finalModel)) #51

outlierTest(ridgeTest1)
################Lasso
xTestLasso <- model.matrix(RPE ~ `Average Metabolic Power` + `HML Efforts` + Decelerations + Accelerations +
                             `Dynamic Stress Load` + Distance, res$test)[,-1]
yTestLasso <- res$test$RPE
glmnet(xTestLasso, yTestLasso, alpha = 1, lambda = NULL)
cvTest <- cv.glmnet(xTestLasso, yTestLasso, alpha = 1)
# Display the best lambda value
cvTest$lambda.min
# Fit the final model on the test data
lassoTest <- glmnet(xTestLasso, yTestLasso, alpha = 1, lambda = cvTest$lambda.min)
# Dsiplay regression coefficients
coef(lassoTest)

lassoTest1 <- train(
  RPE ~ `Average Metabolic Power` + `HML Efforts` + Decelerations + Accelerations +
    `Dynamic Stress Load` + Distance, data = res$test, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda)
)

lassoTest1
RMSE(lassoTest1) #1.428043
count6 <- summary(influence.measures(lassoTest1)) #42
outlierTest(lassoTest1)
################Elastic Net
# Build the model using the test set
set.seed(42)
elasticTest <- train(
  RPE ~ `Average Metabolic Power` + `HML Efforts` + Decelerations + Accelerations +
    `Dynamic Stress Load` + `Speed Intensity` + Distance + `HML Distance` + 
    Duration + `Sprint Distance`, data = res$test, method = "glmnet",
  trControl = trainControl("cv", number = 5),
  tuneLength = 10
)
# Best tuning parameter
elasticTest$bestTune
elasticTest
# Coefficient of the final model. You need to specify the best lambda
coef(elasticTest$finalModel, elasticTest$bestTune$lambda)

RMSE(elasticTest) #1.319529
count7 <- summary(influence.measures(elasticTest)) #37
outlierTest(elasticTest)
########SVM
OptModelsvmTest <- tune(svm, RPE ~ `Average Metabolic Power` + Distance + 
                          `Speed Intensity` + `HML Distance` + `Dynamic Stress Load` + 
                          `HML Efforts` + Decelerations + Sprints + Accelerations +
                          Impacts, data=res$test,ranges=list(elsilon=seq(0,1,0.1), cost=1:100))
OptModelsvmTest
bestSVMTest <- OptModelsvmTest$best.model
bestSVMTest
RMSE(bestSVMTest) #1.128293
count8 <- summary(influence.measures(svm.model)) #33
outlierTest(svm.model)
########Boosted Forests
set.seed(42)
boostedTest <- train(
  RPE ~ `HML Distance` + `Average Metabolic Power` + Accelerations + Duration + `Speed Intensity` +
    Decelerations + `Sprint Distance` + `Dynamic Stress Load`, data = res$test, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
RMSE(boostedTest) #0.8159172
count9 <- summary(influence.measures(boostedTest)) #36
outlierTest(boostedTest)


######################################## SVM has Best Model
OptModelsvmValidate <- tune(svm, RPE ~ `Average Metabolic Power` + Distance + 
                              `Speed Intensity` + `HML Distance` + `Dynamic Stress Load` + 
                              `HML Efforts` + Decelerations + Sprints + Accelerations +
                              Impacts, data=res$validate,ranges=list(elsilon=seq(0,1,0.1), cost=1:100))
OptModelsvmValidate
bestSVMTValidate <- OptModelsvmValidate$best.model
bestSVMTValidate
RMSE(bestSVMTValidate) #1.128293 to 1.074248
count11 <- summary(influence.measures(bestSVMTValidate)) #32
outlierTest(bestSVMTValidate)
########KNN
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
train.loanVal <- res$validate[c(1,11,3,15,5,12,6,10,8,9,16)] # 70% training data
test.loanVal <- res$test[c(1,11,3,15,5,12,6,10,8,9,16)] # remaining 30% test data
#Creating seperate dataframe for rpe feature which is the target.
train.loan_labelsVal <- res$validate[,1]
train.loan_labelsVal
test.loan_labelsVal <-res$test[,1]
test.loan_labelsVal
#Find the number of observation
NROW(train.loan_labelsVal) #17.83, rows 318
set.seed(42)
knnVal.17 <- knn(train=train.loanVal, test=test.loanVal, cl=train.loan_labelsVal, k=17)
set.seed(42)
knnVal.18 <- knn(train=train.loanVal, test=test.loanVal, cl=train.loan_labelsVal, k=18)
#Calculate the proportion of correct classification for k = 17, 18
set.seed(42)
ACCval.17 <- 100 * sum(test.loan_labelsVal == knnVal.17)/NROW(test.loan_labelsVal)
set.seed(42)
ACCval.18 <- 100 * sum(test.loan_labelsVal == knnVal.18)/NROW(test.loan_labelsVal)
ACCval.17
ACCval.18
# Check prediction against actual value in tabular form for k=17
table(knnVal.17 ,test.loan_labelsVal)
cor(as.numeric(knnVal.17), test.loan_labelsVal)
# Check prediction against actual value in tabular form for k=18
table(knnVal.18 ,test.loan_labelsVal)
cor(as.numeric(knnVal.18), test.loan_labelsVal)

uKNNval <- union(knnVal.18, res$validate$RPE)
tKNNval <- table(factor(knnVal.18, uKNNval), factor(res$validate$RPE, uKNNval))
confusionMatrix(tKNNval)
#optimizing the accuracy
i=1
k.optm=1
set.seed(42)
for (i in 1:75){
  knn.modVal <- knn(train=train.loanVal, test=test.loanVal,cl=train.loan_labelsVal, k=i)
  k.optm[i] <- 100 * sum(test.loan_labelsVal == knn.modVal)/NROW(test.loan_labelsVal)
  k=i
  cat(k,'=',k.optm[i],'
      ')
}
#best is k = 30, acc = 31.44654
uKNNval <- union(knnVal.18, res$validate$RPE)
tKNNval <- table(factor(knnVal.18, uKNNval), factor(res$validate$RPE, uKNNval))
confusionMatrix(tKNNval)
########################################

dataEx <- data.frame(col1=runif(20), col2=runif(20), 
                     col3=runif(20), col4=runif(20), col5=runif(20))
bootControl <- trainControl(number = 1)
knnGrid <- expand.grid(.k=c(2:5))
set.seed(42)
knnFit1 <- train(dataEx[,-c(1)], dataEx[,1], method = "knn", trControl = bootControl, 
                 verbose = FALSE, tuneGrid = knnGrid )
knnFit1 

knnFit1$results
knnFit1$bestTune

knnFit1$finalModel

knnFit1$results$Rsquared
knnFit1.sorted <- knnFit1$results[order(knnFit1$results$Rsquared),]
knnFit1.sorted
knnFit1.sorted[1,'Rsquared']



#New column id 

#all possoble regression
#press, r^2


lmTest <- MSpred(RPE ~ Distance + `Max Speed` + `HML Distance` + `HML Efforts` + `Sprint Distance`, data = testData)
lmTest2 #<- MSpred(other models)
tset3

lmValidate <- MSpred(test2, data = valiate)

kTrain <- ols_step_forward_aic(lmTrain)
kTrain

kTest <- ols_step_forward_aic(lmTest) #avg metabolic, avg HR, and decelerations
kTest

kNew <- ols_step_forward_aic(lm1)
kNew


#First Level: Training Data
##All possible regressions
allReg <- ols_step_all_possible(lmTrain)
plot(allReg)
allReg
allReg$model
##Compare in terms of PRESS
PRESS1 <- function(linear.model) {
  #' calculate the predictive residuals
  pr <- residuals(linear.model)/(1-lm.influence(linear.model)$hat)
  #' calculate the PRESS
  PRESS <- sum(pr^2)
  
  return(PRESS)
}

PRESS(allReg) #3571, AIC() = 6041, BIC() = 6150 #######For all possiblle regressions

#Sort based on adj r^2, cp, etc. for vars

#Based on adj R^2
#The top6 regressors are: avg metabolic power, speed intensity, distance, hml distance, decelerations, hml efforts

#Second Level:
################################################
#Splitting into 3 datasets
splits = c(train = .6, test = .2, validate = .2)
################################################
#or should all possible be using train
lmTrain <- lm(RPE ~ Distance + `Max Speed` + `HML Distance` + `HML Efforts` + `Sprint Distance` + Sprints + 
                Accelerations + Decelerations + `Average Heart Rate` + `Max Heart Rate` + 
                `Average Metabolic Power` + `Dynamic Stress Load` + `Heart Rate Exertion` + 
                `High Speed Running (Relative)` + `HML Density` + `Speed Intensity` + 
                Impacts + Duration, data = res$train)
#also test it to compare
################################################

forward1 <- ols_step_forward_aic(RPE ~ all 19 models, data = res$train) #with all other models with all k's
forward1 #avg metabolic, deceleration, hml density, avg hr, distance, max HR  off of AIC
plot(forward1)
forward1$model
#based on adj R^2, AIC, malloc cp, bic

##not yetmod_error(forward1) #comparing the MSpred for the three diffferent models from forward AIC
#not yet Then take model with best MSpred, and use on validate



#Repeat same steps using all 19 regressors for train data for: 2 model cp / adj r2
##Backwards AIC
##Stepwise AIC
##KNN 
##Boosted Forests
##Normal GLM
##Poisson Regression
##Penalized Regression
##Ridge
##Lasso
##Elastic Net
##SVM


#Then do top k for each approach, use on test
#MSpred and problem children - combined

#avg metabolic, deceleration, hml density

#Winner - KNN with its k Regressors and use that on validation dataset
#MSpred and problem children - how is it changing, confirming if similar






###Where each has a different model of regressors
###And the best model is chosen by best MSpred, also dealing with problem children
###Then take that best model and use it against the validation data set

#Compute MSpred of each validation for each of the best models
#Compare which approach has the MSpred, 
#Conclusion: that model and approach is best for predictions
