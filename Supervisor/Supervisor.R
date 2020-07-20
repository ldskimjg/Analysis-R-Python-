library(tidyverse)
library(car)
library(GGally)
library(MASS)

# Read data
supervisor <- read.table(file = "https://mheaton.byu.edu/Courses/Stat330/InClassAnalyses/4%20-%20SupervisorPerformance/Data/Supervisor.txt", sep = "", header = TRUE)

# Build Matrix scatter plots
ggpairs(supervisor)

# Fitting model
mlr <- lm(Rating ~ ., data = supervisor)
summary(mlr)

# Std.Residual histogram
std <- stdres(mlr)
hist(std)

# Added variable plots
avPlots(mlr)

# Residual plots
plot(mlr$residuals ~ mlr$fitted.values)
abline(h=0)

# Try prediction
predict.lm (mlr,newdata=data.frame(Complaints = 65, Privileges = 51.5, Learn = 56.5, Raises = 63.5, Critical = 77.5, Advance = 41), interval="prediction", level = 0.95)

# Cross Validation
n.cv <- 250
bias <- rep(NA,n.cv)
rpmse <- rep(NA,n.cv)
cvg <- rep(NA,n.cv)
wid <- rep(NA,n.cv)
n.test <- nrow(supervisor)/10

for(i in 1:n.cv){
  #split into test and training set
  test.obs <- sample(1:nrow(supervisor),n.test)
  test.set <- supervisor[test.obs,]
  train.set <- supervisor[-test.obs,]
  
  # fit a lm using training data only
  train.lm <- lm(Rating ~ ., data=train.set)
  
  # Prediction and prediction intervals
  test.pred <- predict.lm(train.lm,newdata = test.set,interval="prediction",level = 0.95)  
  
  # calculate results
  bias[i] <- mean(test.pred[,1] - test.set$Rating)
  rpmse[i] <- sqrt(mean((test.pred[,1] - test.set$Rating)^2))
  cvg[i] <- mean(test.pred[,2] < test.set$Rating & test.pred[,3] > test.set$Rating)
  wid[i] <- mean(test.pred[,3]-test.pred[,2])  
}
mean(bias)
mean(rpmse)
mean(cvg)
mean(wid)

