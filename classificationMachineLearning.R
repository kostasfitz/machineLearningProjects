#Machine Learning, Classification methods: kNN, Trees, logistic Regression


data <- read.delim(choose.files(), header=T)

# dependencies
library(ISLR)
library(class)
library(tree)
library(caret)
library(boot)
library(car)

#==================================
# PREPROCESSING DATA 

names(data)
summary(data$effort)
attach(data)
# data normalization
y.values = effort
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
x.norm <- as.data.frame(lapply(data[,2:10], min_max_norm))
detach(data)

# making class variable categorical
y.mean = mean(y.values)
y.min = min(y.values)
y.max = max(y.values)
y.values = cut(y.values, breaks=c(y.min,y.mean,y.max), labels = c("bad", "good"),include.lowest=TRUE)

# combining categorical y with normalized predictors data frame
data.mod = cbind(y.values, x.norm)
names(data.mod)[1] <- "effort"

# splitting data in train and test sets
set.seed(2) 
index <- sample(1:nrow(data.mod), round(nrow(data.mod) * 0.7))
train.x = data.mod[index,-1]
test.x = data.mod[-index,-1]
train.y = data.mod$effort[index]
test.y = data.mod$effort[-index]

train = data.mod[index,]
test = data.mod[-index,]
#==================================

#==================================
# kNN CLASSIFIER 

set.seed(2)
#cv for k 1-15
trControl <- trainControl(method  = "cv",
                          number  = 10)
cv.knn <- train(effort ~ .,
                method     = "knn",
                tuneGrid   = expand.grid(k = 1:15),
                trControl  = trControl,
                metric     = "Accuracy",
                data       = train)

plot(cv.knn)
k.best = cv.knn$bestTune
k.best

knn.pred=knn(train.x,test.x,train.y,k=k.best)
table(Predicted=knn.pred, Actual=test.y)
(114+14)/(152)
mean(knn.pred != test.y) # error rate based on test set

#bootstrap test 
boot.fn=function(data.boot, i){
  sample = data.boot[i, ]
  pred=knn(train.x,sample[,-1],train.y,k=k.best)
  error.rate=mean(pred != sample[,1])
  return(error.rate)
}

set.seed(19)
b.knn=boot(data.mod,boot.fn,1000)
plot(b.knn)
b.knn
mean(b.knn$t)
boot.ci(b.knn,type="norm")
#==================================


#===================================
# CLASSIFICATION TREES 
attach(data.mod)
# fitting model
tree.model = tree(train.y~., train.x)
plot(tree.model)
text(tree.model, pretty=0)

# testing model with test set
tree.pred = predict(tree.model, test.x, type="class")
mean(tree.pred != test.y) # misclassification error
table(Predicted=tree.pred, Actual=test.y)
(89+24)/(89+24+27+12)

#cv for pruning parameters
set.seed(3)
tree.cv = cv.tree(tree.model, FUN=prune.misclass)
names(tree.cv)
plot(tree.cv$size, tree.cv$var, type="b")

# pruning tree
tree.pruned = prune.misclass(tree.model, best=5)
plot(tree.pruned)
text(tree.pruned, pretty=0)

tree.pred2 = predict(tree.pruned, test.x, type="class")
mean(tree.pred2 != test.y) #misclassification error
table(Predicted=tree.pred2, Actual=test.y)
mean(tree.pred != test.y)

#bootstrap test 
boot.fn=function(data.boot, i){
  sample = data.boot[i, ]
  pred = predict(tree.pruned, sample[, -1], type="class")
  error.rate=mean(pred != sample[, 1])
  return(error.rate)
}

set.seed(19)
b.tree=boot(data.mod,boot.fn,1000)
b.tree
plot(b.tree)
mean(b.tree$t)
boot.ci(b.tree,type="norm")

#================================ 
# LOGISTIC REGRESSION
# assumptions: *
cor.x = cor(data.mod[,-1]) # no multicollinearity
glm.data =subset(data.mod,select=-c(Deletedcount,AdjustedFunctionPoints,
                                    Addedcount,Filecount,Interfacecount,Changedcount)) # unwanted predictors 

# fit model
attach(data.mod)
glm.fit=glm(effort~.,data=glm.data[index,], family=binomial)
summary(glm.fit)
coef(glm.fit)
summary(glm.fit)$coef
vif(glm.fit)
# test model with test set
glm.probs=predict(glm.fit, glm.data[-index,], type="response")
glm.probs[1:10]
glm.pred=rep("bad",152)
glm.pred[glm.probs>.5]="Up"
table(Predicted=glm.pred, Actual=test.y)
mean(glm.pred != test.y)

# evaluate if model is statistically significant 
with(glm.fit, pchisq(null.deviance - deviance, df.null-df.residual, lower.tail = F))

# linearity of independent variables and log odds*
plot(train$Outputcount, glm.fit$fitted.values, ylab="logOdds")
plot(train$Inputcount, glm.fit$fitted.values, ylab="logOdds")
plot(train$Enquirycount, glm.fit$fitted.values, ylab="logOdds")
plot(train$Changedcount, glm.fit$fitted.values, ylab="logOdds")
plot(train$Interfacecount, glm.fit$fitted.values, ylab="logOdds")

#bootstrap test 
boot.fn=function(data.boot, i){
  sample = data.boot[i, ]
  probs=predict(glm.fit, sample, type="response")
  pred=rep("bad",152)
  pred[glm.probs>.5]="Up"
  error.rate=mean(pred != sample[,1])
  return(error.rate)
}
set.seed(4)
b.glm=boot(test,boot.fn,1000)
b.glm
plot(b.glm)
mean(b.glm$t)
boot.ci(b.glm, type="norm")
