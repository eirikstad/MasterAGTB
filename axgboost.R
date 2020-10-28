# Comparisons on ISLR data
# Simulated splits: distributions
# Normalized to mean XGB = 1

devtools::install_github("ja-thomas/autoxgboost")

library(MASS)
library(ISLR)
#library(ElemStatLearn)
library(tree)
library(randomForest)
library(xgboost)
library(gbm)
library(ggplot2)
#library(gbtorch)
library(mlr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
load("res_list2.RData")

# datasets
dataset <- function(i, seed)
{
  
  # Returns training and test datasets for x and y
  # i = 1:7 returns mse datasets
  # i = 8:12 returns logloss datasets
  # i = 1: Boston
  # i = 2: ozone
  # i = 3: Auto
  # i = 4: Carseats
  # i = 5: College
  # i = 6: Hitters
  # i = 7: Wage
  # i = 8: Caravan
  # i = 9: Default
  # i = 10: OJ
  # i = 11: Smarket
  # i = 12: Weekly
  
  # reproducibility
  set.seed(seed) 
  
  if(i==1){
    # boston
    data(Boston)
    medv_col <- which(colnames(Boston)=="medv")
    n_full <- nrow(Boston)
    ind_train <- sample(n_full, 0.5*n_full)
    x.train <<- as.matrix(Boston[ind_train,-medv_col], dimnames = make.names(colnames(Boston)))
    y.train <<- as.matrix(log(Boston[ind_train,medv_col]))
    x.test <<- as.matrix(Boston[-ind_train,-medv_col])
    y.test <<- as.matrix(log(Boston[-ind_train,medv_col]))
    colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
    colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
    
  }else 
    if(i==2){
      # ozone
      data(ozone)
      n_full <- nrow(ozone)
      ind_train <- sample(n_full, 0.5*n_full)
      x.train <<- as.matrix(log(ozone[ind_train,-1]), dimnames = make.names(colnames(ozone)))
      y.train <<- as.matrix(log(ozone[ind_train,1]))
      x.test <<- as.matrix(log(ozone[-ind_train,-1]))
      y.test <<- as.matrix(log(ozone[-ind_train,1]))
      colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
      colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
      
    }else 
      if(i==3){
        
        # auto
        data("Auto")
        dim(Auto)
        mpg_col <- which(colnames(Auto)=="mpg")
        n_full <- nrow(Auto)
        data <- model.matrix(mpg~.,data=Auto)[,-1]
        dim(data)
        ind_train <- sample(n_full, 0.5*n_full)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(Auto[ind_train, mpg_col])
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(Auto[-ind_train, mpg_col])
        colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
        colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
      }else 
        if(i==4){
          # Carseats - sales - mse
          data("Carseats")
          dim(Carseats)
          Carseats =na.omit(Carseats)
          dim(Carseats)
          sales_col <- which(colnames(Carseats)=="Sales")
          n_full <- nrow(Carseats)
          data <- model.matrix(Sales~., data=Carseats)[,-1]
          dim(data)
          ind_train <- sample(n_full, 0.7*n_full)
          x.train <<- as.matrix(data[ind_train, ])
          y.train <<- as.matrix(Carseats[ind_train, sales_col])
          x.test <<- as.matrix(data[-ind_train, ])
          y.test <<- as.matrix(Carseats[-ind_train, sales_col])
          colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
          colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
          
        }else 
          if(i==5){
            
            # College - apps: applications received - mse
            data("College")
            dim(College)
            n_full <- nrow(College)
            ind_train <- sample(n_full, 0.7*n_full)
            data <- model.matrix(Apps~., data=College)[,-1]
            dim(data)
            x.train <<- as.matrix(data[ind_train, ])
            y.train <<- as.matrix(College[ind_train, "Apps"])
            x.test <<- as.matrix(data[-ind_train, ])
            y.test <<- as.matrix(College[-ind_train, "Apps"])
            colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
            colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
            
          }else 
            if(i==6){
              # Hitters: Salary - mse
              data("Hitters")
              dim(Hitters)
              Hitters =na.omit(Hitters)
              dim(Hitters)
              n_full <- nrow(Hitters)
              ind_train <- sample(n_full, 0.7*n_full)
              data <- model.matrix(Salary~., data=Hitters)[,-1]
              dim(data)
              x.train <<- as.matrix(data[ind_train, ])
              y.train <<- as.matrix(Hitters[ind_train, "Salary"])
              x.test <<- as.matrix(data[-ind_train, ])
              y.test <<- as.matrix(Hitters[-ind_train, "Salary"])
              colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
              colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
              
            }else 
              if(i==7){
                # Wage - Wage - mse -- note: extremely deep trees!
                data(Wage)
                dim(Wage)
                n_full <- nrow(Wage)
                ind_train <- sample(n_full, 0.7*n_full)
                data <- model.matrix(wage~., data=Wage)[,-1]
                dim(data)
                x.train <<- as.matrix(data[ind_train, ])
                y.train <<- as.matrix(Wage[ind_train, "wage"])
                x.test <<- as.matrix(data[-ind_train, ])
                y.test <<- as.matrix(Wage[-ind_train, "wage"])
                colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
                colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
                
              }else 
                if(i==8){
                  # Caravan - classification
                  data(Caravan)
                  dim(Caravan)
                  Caravan = na.omit(Caravan)
                  dim(Caravan)
                  n_full <- nrow(Caravan)
                  ind_train <- sample(n_full, 0.7*n_full)
                  data <- model.matrix(Purchase~., data=Caravan)[,-1]
                  dim(data)
                  x.train <<- as.matrix(data[ind_train, ])
                  y.train <<- as.matrix(ifelse(Caravan[ind_train, "Purchase"]=="Yes",1,0))
                  x.test <<- as.matrix(data[-ind_train, ])
                  y.test <<- as.matrix(ifelse(Caravan[-ind_train, "Purchase"]=="Yes", 1, 0))
                  colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
                  colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
                  
                }else 
                  if(i==9){
                    # Default - Default: if default on credit - classification
                    data("Default")
                    dim(Default)
                    n_full <- nrow(Default)
                    ind_train <- sample(n_full, 0.7*n_full)
                    data <- model.matrix(default~., data=Default)[,-1]
                    dim(data)
                    x.train <<- as.matrix(data[ind_train, ])
                    y.train <<- as.matrix(ifelse(Default[ind_train, "default"]=="Yes",1,0))
                    x.test <<- as.matrix(data[-ind_train, ])
                    y.test <<- as.matrix(ifelse(Default[-ind_train, "default"]=="Yes", 1, 0))
                    colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
                    colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
                    
                  }else 
                    if(i==10){
                      # OJ: Purchase - Classification
                      data("OJ")
                      dim(OJ)
                      n_full <- nrow(OJ)
                      ind_train <- sample(n_full, 0.7*n_full)
                      data <- model.matrix(Purchase~., data=OJ)[,-1]
                      dim(data)
                      x.train <<- as.matrix(data[ind_train, ])
                      y.train <<- as.matrix(ifelse(OJ[ind_train, "Purchase"]=="MM",1,0))
                      x.test <<- as.matrix(data[-ind_train, ])
                      y.test <<- as.matrix(ifelse(OJ[-ind_train, "Purchase"]=="MM", 1, 0))
                      colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
                      colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
                      
                    }else 
                      if(i==11){
                        # Smarket : classification
                        data("Smarket")
                        dim(Smarket)
                        Smarket <- subset(Smarket, select=-c(Today, Year))
                        n_full <- nrow(Smarket)
                        ind_train <- sample(n_full, 0.7*n_full)
                        data <- model.matrix(Direction~., data=Smarket)[,-1]
                        dim(data)
                        x.train <<- as.matrix(data[ind_train, ])
                        y.train <<- as.matrix(ifelse(Smarket[ind_train, "Direction"]=="Up",1,0))
                        x.test <<- as.matrix(data[-ind_train, ])
                        y.test <<- as.matrix(ifelse(Smarket[-ind_train, "Direction"]=="Up", 1, 0))
                        colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
                        colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
                        
                      }else 
                        if(i==12){
                          # Weekly - classification
                          data("Weekly")
                          dim(Weekly)
                          Weekly <- subset(Weekly, select=-c(Today, Year))
                          n_full <- nrow(Weekly)
                          ind_train <- sample(n_full, 0.7*n_full)
                          data <- model.matrix(Direction~., data=Weekly)[,-1]
                          dim(data)
                          x.train <<- as.matrix(data[ind_train, ])
                          y.train <<- as.matrix(ifelse(Weekly[ind_train, "Direction"]=="Up",1,0))
                          x.test <<- as.matrix(data[-ind_train, ])
                          y.test <<- as.matrix(ifelse(Weekly[-ind_train, "Direction"]=="Up", 1, 0))
                          colnames(x.train) = make.names(colnames(x.train), unique = TRUE)
                          colnames(x.test) = make.names(colnames(x.test), unique = TRUE)
                          
                        }
}


# loss function
loss_mse <- function(y,y.hat){
  sum((y-y.hat)^2)
}

LogLossBinary = function(y, x) {
  x = pmin(pmax(x, 1e-14), 1-1e-14)
  -sum(y*log(x) + (1-y)*log(1-x)) / length(y)
}



myparset = autoxgboost::autoxgbparset


myparset[["pars"]][["lambda"]][["trafo"]] = function(x) 0
myparset[["pars"]][["lambda"]][["lower"]] = 0
myparset[["pars"]][["lambda"]][["upper"]] = 1


#myparset[["pars"]][["colsample_bytree"]][["lower"]] =1
#myparset[["pars"]][["colsample_bytree"]][["tunable"]] = FALSE

#myparset[["pars"]][["colsample_bylevel"]][["lower"]] = 1
#myparset[["pars"]][["colsample_bylevel"]][["tunable"]] = FALSE


myparset[["pars"]][["alpha"]][["lower"]] = 0
myparset[["pars"]][["alpha"]][["upper"]] = 1
myparset[["pars"]][["alpha"]][["trafo"]] = function(x) 0

myparset[["pars"]][["subsample"]][["trafo"]] = function(x) 1




#Autoxgboosttest


# Seeds
options( warn = -1 )
set.seed(14912)
B <- 100
seeds <- sample(1e5, B) 
S = 1
# Store results
for(i in c(4:7)){
  cat("iter: ", i,"\n")
  res_mat <- matrix(nrow=B, ncol=S)
  for (b in 1:B){
    for(s in 1:S){
      
      # generate data
      dataset(i, seeds[b])
    
      
      #autoxgb
      axgbdata = cbind(y.train, x.train)
      axgbtask = mlr::makeRegrTask(data = as.data.frame(axgbdata), target = 'V1')
      axgbmod = autoxgboost::autoxgboost(axgbtask, time.budget = 15*s, build.final.model = TRUE, par.set = myparset)
      
      
      axgb.pred = predict(axgbmod$final.model, newdata = as.data.frame(x.test))$data$response
    
      
      # update res matrice
      res_mat[b, s] <- loss_mse(y.test, axgb.pred)
    }
  }
  res[[i]] = cbind(res[[i]], res_mat)
}


mxgb <- sapply(1:7, function(i) mean(res[[i]][,6]))
datasets <- c("Boston", "Ozone", "Auto", "Carseats", "College", "Hitters", "Wage")
order <- c(6,1,5,4,3,2,7)

for(i in 1:7){
  cat(datasets[i], " & ")
  for(j in 1:7){
    cat( paste0(
      format(  mean(res[[i]][ , order[j]] / mxgb[i]) , digits=3 ), " (", 
      format(  sd(res[[i]][ , order[j]]/ mxgb[i]) , digits=3 ), ")  & "
    ) )
  }
  cat("\\\\ ","\n")
}
cat("\\hline \n")




for(i in c(8:12)){
  cat("iter: ", i,"\n")
  res_mat <- matrix(nrow=B, ncol=S)
  for (b in 1:B){
    for(s in 1:S){
      
      # generate data
      dataset(i, seeds[b])
      
      
      #autoxgb
      axgbdata = as.data.frame(cbind(y.train, x.train))
      axgbdata$V1 = as.factor(axgbdata$V1)
      axgbtask = mlr::makeClassifTask(data = as.data.frame(axgbdata), target = 'V1')
      axgbmod = autoxgboost::autoxgboost(axgbtask, iterations = 15*s, build.final.model = TRUE, par.set = myparset)
      
      
      axgb.pred = as.numeric(predict(axgbmod$final.model, newdata = as.data.frame(x.test))$data$response)
      
      
      # update res matrice
      res_mat[b, s] <- LogLossBinary(y.test, axgb.pred)
      
    }
  }
  res[[i]] = cbind(res[[i]], res_mat)
}





mxgb <- sapply(1:12, function(i) mean(res[[i]][,4]))
datasets <- c("Boston", "Ozone", "Auto", "Carseats", "College", "Hitters", "Wage",
              "Caravan", "Default", "OJ", "Smarket", "Weekly")
order2 <- c(4,1,3,2)
for(i in 8:12){
  cat(datasets[i], " & ")
  for(j in 1:5){
    cat( paste0(
      format(  mean(res[[i]][ , order2[j]] / mxgb[i]) , digits=3 ), " (", 
      format(  sd(res[[i]][ , order2[j]]/ mxgb[i]) , digits=3 ), ")  & "
    ) )
  }
  cat("\\\\ ","\n")
}


