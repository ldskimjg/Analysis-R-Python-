library(tidyverse)
library(car)
library(GGally)
library(MASS)

# Read data
bodyfat <- read.table(file = "BodyFat.txt", sep = "", header = TRUE)

# Build Matrix scatter plots
ggpairs(bodyfat)
cov(bodyfat)
cor(bodyfat)

# Fitting model
mlr <- lm(brozek ~ ., data = bodyfat)
summary(mlr)

# Std.Residual histogram
std <- stdres(mlr)
hist(std)

# Added variable plots
avPlots(mlr)

# Residual plots
plot(mlr$residuals ~ mlr$fitted.values)
abline(h=0)

# Cross Validation
n.cv <- 250
bias <- rep(NA,n.cv)
rpmse <- rep(NA,n.cv)
cvg <- rep(NA,n.cv)
wid <- rep(NA,n.cv)
n.test <- 25

for(i in 1:n.cv){
  #split into test and training set
  test.obs <- sample(1:nrow(bodyfat),n.test)
  test.set <- bodyfat[test.obs,]
  train.set <- bodyfat[-test.obs,]
  
  # fit a lm using training data only
  train.lm <- lm(brozek ~ ., data=train.set)
  
  # Prediction and prediction intervals
  test.pred <- predict.lm(train.lm,newdata = test.set,interval="prediction",level = 0.95)  
  
  # calculate results
  bias[i] <- mean(test.pred[,1] - test.set$brozek)
  rpmse[i] <- sqrt(mean((test.pred[,1] - test.set$brozek)^2))
  cvg[i] <- mean(test.pred[,2] < test.set$brozek & test.pred[,3] > test.set$brozek)
  wid[i] <- mean(test.pred[,3]-test.pred[,2])  
}
mean(bias)
mean(rpmse)
mean(cvg)
mean(wid)

# Try prediction
predict.lm (mlr,newdata=data.frame(age = 50, weight = 203, height = 67, neck = 40.2, chest = 114.8, abdom = 108.1, hip = 102.5, thigh = 61.3, knee = 41.1, ankle = 24.7, biceps = 34.1, forearm = 31, wrist = 18.3), interval="prediction", level = 0.95)
