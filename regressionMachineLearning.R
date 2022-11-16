#Machine Learning, Regression methods

data <- read.delim(choose.files(), header=T)

library(ISLR)
library(boot)
library(pls)
library(glmnet)
library(car)

#standardize data
standata = scale(data, center=TRUE, scale=TRUE)
data.all = data.frame(standata)

attach(data.all)
names(data.all)


#LINEAR REGRESSION

#linear relationship between y and x check
par(mfrow=c(1,1))
plot(AdjustedFunctionPoints,effort)
plot(Inputcount,effort, xlim=c(-0.2,4.5))
plot(Enquirycount,effort)
plot(Filecount,effort)
plot(Interfacecount,effort, xlim=c(-0.5,5))
plot(Outputcount,effort, xlim=c(-0.5,3))
plot(Addedcount,effort, xlim=c(-0.5,5))
plot(Changedcount,effort, xlim=c(-0.5,5))
plot(Deletedcount,effort, xlim=c(-0.5,3))
#multicollinearity check
cor.matrix=cor(data.all[2:10])

#rejecting variables
detach(data.all)
dataset=subset(data.all, select=-c(Deletedcount,AdjustedFunctionPoints,Addedcount,Filecount))
cor.matrix=cor(dataset[,2:6])
attach(dataset)

#data split 
set.seed(1) 
index <- sample(1:nrow(dataset), round(nrow(dataset) * 0.7))
train <- dataset[index, ]
test  <- dataset[-index, ]

#stats of y in test set
summary(test$effort)

#multiple linear regression model fitting 
lm.fit=lm(effort~.,data=train)
summary(lm.fit)
plot(lm.fit)

#linear reg assumptions check
#constant variance of residuals
plot(lm.fit$fitted.values, lm.fit$residuals)
#no autocorrellation 
plot(lm.fit$residuals)

#Testing model
#test multiple linear regression with testset
lm.predictions = predict.lm(lm.fit, test)
summary(lm.predictions)
mse.lm = mean((test$effort - lm.predictions) ^ 2)
mse.lm

#bootstrap test 
boot.fn=function(data.boot, i){
  sample = data.boot[i, ]
  pred=predict.lm(lm.fit, sample)
  mse = mean((sample$effort-pred)^2)
  return(mse)
}
set.seed(19)
b=boot(test,boot.fn,5000)
b
boot.ci(b,type="norm")



#PRINCIPAL COMPONENTS REGRESSION
detach(dataset)
attach(data.all)
#data split 
set.seed(1) 
index <- sample(1:nrow(data.all), round(nrow(data.all) * 0.7))
train.all <- data.all[index, ]
test.all <- data.all[-index, ]

#principal components regression cross valid
set.seed(1)
pcr.fit=pcr(effort~., data=train.all,scale=TRUE,validation="CV")
summary(pcr.fit)
validationplot(pcr.fit,val.type="MSEP")
plot(pcr.fit$Xvar)

#Testing model
#predictions and mse calculation
pcr.pred=predict(pcr.fit,test.all,ncomp=4)
summary(pcr.pred)
mse.pcr = mean((test.all$effort - pcr.pred)^2)
mse.pcr

#bootstrap test pcr
boot.fn2=function(data.boot, i){
  sample = data.boot[i, ]
  pred=predict(pcr.fit, sample)
  mse = mean((sample$effort-pred)^2)
  return(mse)
}
set.seed(19)
b.pcr=boot(test.all,boot.fn2,5000)
b.pcr
boot.ci(b.pcr,type="norm")



#RIDGE REGRESSION

x.train=model.matrix(effort~.,train.all)[,-1]
y.train=train.all$effort
x.test=model.matrix(effort~.,test.all)[,-1]
y.test=test.all$effort

ridge.mod=glmnet(x.train,y.train,alpha=0)
#cross val to find best lambda
set.seed(1)
cv.out=cv.glmnet(x.train,y.train,alpha=0)
plot(cv.out)
cv.out$glmnet.fit$lambda
bestlam=cv.out$lambda.min
bestlam
ridge.bestmod = glmnet(x.train,y.train,alpha=0, lambda=bestlam)
coef(ridge.bestmod)

#Testing ridge model
#predictions and mse calculation
ridge.pred=predict(ridge.bestmod,s=bestlam,newx=x.test)
summary(ridge.pred)
mse.ridge = mean((ridge.pred-y.test)^2)
mse.ridge

#bootstrap test ridge (probably has bugs)
boot.fn3=function(data.boot, i){
  sample.x=model.matrix(effort~.,data.boot)[i,-1]
  sample.y=data.boot$effort
  pred=predict(ridge.bestmod, sample.x)
  mse = mean((sample.y-pred)^2)
  return(mse)
}
set.seed(19)
b.ridge=boot(train.all,boot.fn3,5000)
b.ridge
boot.ci(b.ridge,type="norm")





