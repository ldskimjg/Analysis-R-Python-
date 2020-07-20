library(tidyverse)

water <- read.table("https://mheaton.byu.edu/Courses/Stat330/HomeworkAnalyses/3%20-%20WaterRunoff/Data/water.txt",sep = "", header = TRUE)

prec <- water$Precip

runf <- water$Runoff

# Check assumption and fit model
ggplot(water, aes(x=prec, y=runf)) + geom_point() + xlab("Precipitation") + ylab("Water Avilability")

cor(prec,runf)

cor(prec,runf)^2

slr <- lm(runf~prec, data = water)

library(lmtest)

bptest(slr)

library(MASS)

std <- stdres(slr)

hist(std)

ks.test(std,"pnorm")

plot(slr$residuals ~ slr$fitted.values)

abline(h=0)

ggplot(water, aes(x=prec, y=runf)) + geom_point() + geom_smooth(method = "lm", se=FALSE)

summary(slr)

confint(slr,level=.95)

# Add prediction line
ggplot(water,aes(x=prec, y=runf))+
  geom_point() + geom_smooth(method = "lm", se=FALSE)
pred_d <- data.frame(prec =seq(min(prec),max(prec),len = 1000))
pred <- predict.lm(slr, newdata = pred_d, interval = "prediction")
pred <- as.data.frame(pred)
ggplot(water, aes(x=prec,y=runf))+geom_point()+geom_smooth(method="lm",se=FALSE) + 
  geom_line(data=pred,aes(x=pred_d$prec,y=lwr),col="red") + 
  geom_line(data=pred,aes(x=pred_d$prec,y=upr),col="red")

#cross validation
n.cv <- 250
bias <- rep(NA,n.cv)
rpmse <- rep(NA,n.cv)
cvg <- rep(NA,n.cv)
wid <- rep(NA,n.cv)
n.test <- 4

for(i in 1:n.cv){
  #split into test and training set
  test.obs <- sample(1:nrow(water),n.test)
  test.set <- water[test.obs,]
  train.set <- water[-test.obs,]
  
  # fit a lm using training data only
  train.lm <- lm(Runoff ~ Precip, data=train.set)
  
  # Prediction and prediction intervals
  test.pred <- predict.lm(train.lm,newdata = test.set,interval="prediction",level = 0.95)  
  
  # calculate results
  bias[i] <- mean(test.pred[,1] - test.set$Runoff)
  rpmse[i] <- sqrt(mean((test.pred[,1] - test.set$Runoff)^2))
  cvg[i] <- mean(test.pred[,2] < test.set$Runoff & test.pred[,3] > test.set$Runoff)
  wid[i] <- mean(test.pred[,3]-test.pred[,2])  
}
mean(bias)
mean(rpmse)
mean(cvg)
mean(wid)

predict.lm (slr,newdata=data.frame(prec=c(4.5)), interval="prediction", level = 0.95)

