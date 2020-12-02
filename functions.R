






#Loss function
loss_mse <- function(y,y.hat){
  mean((y-y.hat)^2)
}

LogLossBinary = function(y, x) {
  x = pmin(pmax(x, 1e-14), 1-1e-14)
  -sum(y*log(x) + (1-y)*log(1-x)) / length(y)
}

logistic <- function(x){
  res <- exp(x)/(1+exp(x))
  return(res)
}

logit <- function(x){
  res <- log(x/(1-x))
  return(res)
}

#model with sampling. samSize = 1 should give ordinary agt model
#Forced continued learning should be set to TRUE as generalization loss reduction estimator is not functioning
#Returns list containing:
#mod - list of all agt models(trees)
#gen.loss - vector of Estimated generalization loss for all agt models(trees)
#train.loss - vector of training loss at each iteration
#Nleaves - Vector of nleaves of all agt models(trees)
sampling.agt.train <- function(x.train, y.train, learnRate = 0.1, samSize = 1, colsample_bytree = 1, Nrounds, loss_function = "mse", 
                               type = "reg", algorithm = "global_subset", force_continued_learning = FALSE, 
                               bootstrap = FALSE, verbose = 0){
  
  gen.loss = matrix(0,Nrounds+1,1)
  train.loss = matrix(0,Nrounds,1)
  nleaves = matrix(0,Nrounds,1)
  train.pred = matrix(numeric(length(y.train)))
  modlist = list()
  
  i = 1
  
  reduction = TRUE
  
  if (samSize == 1 & colsample_bytree == 1){
    x.train.samp <- x.train
    y.train.samp <-  y.train
  } else{
    row_smp <- floor(samSize * nrow(x.train))
    row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
    x.train.samp <- x.train[row_ind, ]
    y.train.samp <- y.train[row_ind, ]
    if(colsample_bytree != 1){
      col_smp <- floor(colsample_bytree * ncol(x.train))
      col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
      x.train.samp <- x.train.samp[,col_ind]
    }
  }
  
  #Train first tree with initial prediction
  cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, loss_function = loss_function,
                        nrounds = 1, verbose = verbose, greedy_complexities = F, algorithm = algorithm)
  sample_loss <- loss_mse(y.train.samp, predict(cand.mod, x.train.samp))
  full_loss <- loss_mse(y.train, train.pred + predict(cand.mod, x.train))
  gen.loss[1,1] <- cand.mod$estimate_generalization_loss(1) - sample_loss + full_loss
  
  
  while (reduction & i<(Nrounds+1)){
    
    modlist <- c(modlist, list(cand.mod))
    
    nleaves[i,1] <- cand.mod$get_num_leaves()
    
    
    if(type == "reg"){
      if (colsample_bytree == 1){
        train.pred <- train.pred + predict(cand.mod, x.train)
      } else{
        train.pred <- train.pred + predict(cand.mod, x.train[, col_ind])
      }
      train.loss[i,1] = loss_mse(train.pred, y.train)
    } else{
      if(type == "cla"){
        if (colsample_bytree == 1){
          train.pred <- train.pred + logit(predict(camp.mod, x.train))
        } else{
          train.pred <- train.pred + logit(predict(camp.mod, x.train[,col_ind]))
        }
        train.loss[i,1] = LogLossBinary(train.pred, y.train)
      }
    }
    
    if (samSize == 1 & colsample_bytree == 1){
      x.train.samp <- x.train
      y.train.samp <-  y.train
      train.pred.samp <- train.pred
    } else{
      row_smp <- floor(samSize * nrow(x.train))
      row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
      x.train.samp <- x.train[row_ind, ]
      y.train.samp <- y.train[row_ind, ]
      train.pred.samp <- train.pred[row_ind, ]
      if(colsample_bytree != 1){
        col_smp <- floor(colsample_bytree * ncol(x.train))
        col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
        x.train.samp <- x.train.samp[,col_ind]
      }
    }
    
    #Train trees without initial prediction by previous_pred
    cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, 
                          loss_function = loss_function, nrounds = 1,
                          verbose = verbose, greedy_complexities = F, 
                          previous_pred = train.pred.samp,algorithm = algorithm)
    i = i+1
    
    sample_loss <- loss_mse(y.train.samp, train.pred.samp + predict(cand.mod, x.train.samp))
    full_loss <- loss_mse(y.train, train.pred + predict(cand.mod, x.train))
    gen.loss[i,1] <- cand.mod$estimate_generalization_loss(1) - sample_loss + full_loss - train.loss[(i-1), 1]  + gen.loss[(i-1), 1] 
    
    
    
    if(!force_continued_learning){
      reduction = (gen.loss[i,1]-gen.loss[(i-1),1] < 0)  # Need to find this expression
    }
    
  }
  
  gen.loss = gen.loss[1:i, 1]
  nleaves = nleaves[1:(i-1), 1]
  train.loss = train.loss[1:(i-1), 1]
  
  
  mod <- list("mod" = modlist, "gen.loss" = gen.loss, "train.loss" = train.loss, 
              "Nleaves" = nleaves, "Nrounds" = Nrounds, "SmpSize" = samSize)
  
  return (mod)
}



# First try. For each sampled one-tree model, i train a full model on full data,
# with residuals based on the sampled one-tree models ensamble.
# if full.mod[i] performs better than full.mod[i-1], add sam.mod[i] and keep 
# boosting. Very experimental.
first.sampling.agt.train <- function(x.train, y.train, learnRate = 0.1, samSize = 1, colsample_bytree = 1, Nrounds, loss_function = "mse", 
                                     type = "reg", algorithm = "global_subset", force_continued_learning = FALSE, 
                                     bootstrap = FALSE, verbose = 0){
  
  full.gen.loss = matrix(0,Nrounds+1,1)
  gen.loss = matrix(0,Nrounds+1,1)
  train.loss = matrix(0,Nrounds,1)
  nleaves = matrix(0,Nrounds,1)
  train.pred = matrix(numeric(length(y.train)))
  modlist = list()
  
  i = 1
  
  reduction = TRUE
  
  if (samSize == 1 & colsample_bytree == 1){
    x.train.samp <- x.train
    y.train.samp <-  y.train
  } else{
    row_smp <- floor(samSize * nrow(x.train))
    row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
    x.train.samp <- x.train[row_ind, ]
    y.train.samp <- y.train[row_ind, ]
    if(colsample_bytree != 1){
      col_smp <- floor(colsample_bytree * ncol(x.train))
      col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
      x.train.samp <- x.train.samp[,col_ind]
    }
  }
  
  #Train first tree with initial prediction
  cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, loss_function = loss_function,
                        nrounds = 1, verbose = verbose, greedy_complexities = F, algorithm = algorithm)
  sample_loss <- loss_mse(y.train.samp, predict(cand.mod, x.train.samp))
  full_loss <- loss_mse(y.train, train.pred + predict(cand.mod, x.train))
  gen.loss[1,1] <- cand.mod$estimate_generalization_loss(1) - sample_loss + full_loss
  
  
  while (reduction & i<(Nrounds+1)){
    
    modlist <- c(modlist, list(cand.mod))
    
    nleaves[i,1] <- cand.mod$get_num_leaves()
    
    
    if(type == "reg"){
      if (colsample_bytree == 1){
        train.pred <- train.pred + predict(cand.mod, x.train)
      } else{
        train.pred <- train.pred + predict(cand.mod, x.train[, col_ind])
      }
      train.loss[i,1] = loss_mse(train.pred, y.train)
    } else{
      if(type == "cla"){
        if (colsample_bytree == 1){
          train.pred <- train.pred + logit(predict(camp.mod, x.train))
        } else{
          train.pred <- train.pred + logit(predict(camp.mod, x.train[,col_ind]))
        }
        train.loss[i,1] = LogLossBinary(train.pred, y.train)
      }
    }
    
    if (samSize == 1 & colsample_bytree == 1){
      x.train.samp <- x.train
      y.train.samp <-  y.train
      train.pred.samp <- train.pred
    } else{
      row_smp <- floor(samSize * nrow(x.train))
      row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
      x.train.samp <- x.train[row_ind, ]
      y.train.samp <- y.train[row_ind, ]
      train.pred.samp <- train.pred[row_ind, ]
      if(colsample_bytree != 1){
        col_smp <- floor(colsample_bytree * ncol(x.train))
        col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
        x.train.samp <- x.train.samp[,col_ind]
      }
    }
    
    
    
    i = i+1
    
    full.mod <- gbt.train(y.train, x.train, learnRate, loss_function = loss_function,
                          nrounds = 1, verbose = verbose, previous_pred = train.pred, 
                          greedy_complexities = F, algorithm = algorithm)
    full.gen.loss[i,1] <- full.mod$estimate_generalization_loss(1) - train.loss[(i-1),1] + gen.loss[i-1]
    
    
    #Train trees without initial prediction by previous_pred
    cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, 
                          loss_function = loss_function, nrounds = 1,
                          verbose = verbose, greedy_complexities = F, 
                          previous_pred = train.pred.samp,algorithm = algorithm)
    
    
    sample_loss <- loss_mse(y.train.samp, train.pred.samp + predict(cand.mod, x.train.samp))
    full_loss <- loss_mse(y.train, train.pred + predict(cand.mod, x.train))
    gen.loss[i,1] <- cand.mod$estimate_generalization_loss(1)  - sample_loss + full_loss - train.loss[(i-1), 1]  + gen.loss[(i-1), 1] 
    
    
  
    
    
    if(!force_continued_learning & i>2){
      reduction = (full.gen.loss[i,1]-full.gen.loss[(i-1),1] < 0)  # Need to find this expression
    }
    
  }
  
  gen.loss = gen.loss[1:i, 1]
  nleaves = nleaves[1:(i-1), 1]
  train.loss = train.loss[1:(i-1), 1]
  
  
  mod <- list("mod" = modlist, "gen.loss" = gen.loss, "train.loss" = train.loss, 
              "Nleaves" = nleaves, "Nrounds" = Nrounds, "SmpSize" = samSize)
  
  return (mod)
}


#If the minimum gen loss is not in the last k trees, we stop and include all
#trees up until the min
min.sampling.agt.train <- function(x.train, y.train, learnRate = 0.1, samSize = 1, 
                                   colsample_bytree = 1, Nrounds, loss_function = "mse", a = 10,
                                   type = "reg", algorithm = "global_subset", force_continued_learning = FALSE, 
                                   bootstrap = FALSE, verbose = 0){
  
  gen.loss = matrix(0,Nrounds+1,1)
  train.loss = matrix(0,Nrounds,1)
  nleaves = matrix(0,Nrounds,1)
  train.pred = matrix(numeric(length(y.train)))
  modlist = list()
  min_loss = Inf
  
  i = 1
  
  reduction = TRUE
  
  if (samSize == 1 & colsample_bytree == 1){
    x.train.samp <- x.train
    y.train.samp <-  y.train
  } else{
    row_smp <- floor(samSize * nrow(x.train))
    row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
    x.train.samp <- x.train[row_ind, ]
    y.train.samp <- y.train[row_ind, ]
    if(colsample_bytree != 1){
      col_smp <- floor(colsample_bytree * ncol(x.train))
      col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
      x.train.samp <- x.train.samp[,col_ind]
    }
  }
  
  #Train first tree with initial prediction
  cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, loss_function = loss_function,
                        nrounds = 1, verbose = verbose, greedy_complexities = F, algorithm = algorithm)
  sample_loss <- loss_mse(y.train.samp, predict(cand.mod, x.train.samp))
  full_loss <- loss_mse(y.train, train.pred + predict(cand.mod, x.train))
  gen.loss[1,1] <- cand.mod$estimate_generalization_loss(1) - sample_loss + full_loss
  
  
  while (reduction & i<(Nrounds+1)){
    
    modlist <- c(modlist, list(cand.mod))
    
    nleaves[i,1] <- cand.mod$get_num_leaves()
    
    
    if(type == "reg"){
      if (colsample_bytree == 1){
        train.pred <- train.pred + predict(cand.mod, x.train)
      } else{
        train.pred <- train.pred + predict(cand.mod, x.train[, col_ind])
      }
      train.loss[i,1] = loss_mse(train.pred, y.train)
    } else{
      if(type == "cla"){
        if (colsample_bytree == 1){
          train.pred <- train.pred + logit(predict(camp.mod, x.train))
        } else{
          train.pred <- train.pred + logit(predict(camp.mod, x.train[,col_ind]))
        }
        train.loss[i,1] = LogLossBinary(train.pred, y.train)
      }
    }
    
    if (samSize == 1 & colsample_bytree == 1){
      x.train.samp <- x.train
      y.train.samp <-  y.train
      train.pred.samp <- train.pred
    } else{
      row_smp <- floor(samSize * nrow(x.train))
      row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
      x.train.samp <- x.train[row_ind, ]
      y.train.samp <- y.train[row_ind, ]
      train.pred.samp <- train.pred[row_ind, ]
      if(colsample_bytree != 1){
        col_smp <- floor(colsample_bytree * ncol(x.train))
        col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
        x.train.samp <- x.train.samp[,col_ind]
      }
    }
    
    #Train trees without initial prediction by previous_pred
    cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, 
                          loss_function = loss_function, nrounds = 1,
                          verbose = verbose, greedy_complexities = F, 
                          previous_pred = train.pred.samp,algorithm = algorithm)
    i = i+1
    
    sample_loss <- loss_mse(y.train.samp, train.pred.samp + predict(cand.mod, x.train.samp))
    full_loss <- loss_mse(y.train, train.pred + predict(cand.mod, x.train))
    gen.loss[i,1] <- cand.mod$estimate_generalization_loss(1) - sample_loss + full_loss - train.loss[(i-1), 1]  + gen.loss[(i-1), 1] 
    
    
    
    if(!force_continued_learning){
      if(gen.loss[i,1] < min_loss){
        min_loss = gen.loss[i,1]
        min = i
      }
      if(i-min > a){
        reduction = FALSE  # Need to find this expression
      }
      
    }
    
  }
  
  gen.loss = gen.loss[1:i, 1]
  nleaves = nleaves[1:(i-1), 1]
  train.loss = train.loss[1:(i-1), 1]
  
  mod <- list("mod" = modlist, "gen.loss" = gen.loss, "train.loss" = train.loss, 
              "Nleaves" = nleaves, "Nrounds" = Nrounds, "SmpSize" = samSize)
  
  return (mod)
}


# Since the estimated generalization loss for models with sampled training 
# data of size < n is not directly comparable, I propose a new criteria for early
# stopping. For each new tree to an ensamble, I train a two-tree model on the full
# data with the sample model predictions, and if this is able to reduce the
# estimated generalization in the second tree relative to the first tree, 
# we keep training.

full.sampling.agt.train <- function(x.train, y.train, learnRate = 0.1, samSize = 1, colsample_bytree = 1, 
                                    Nrounds = 1000, loss_function = "mse", type = "reg", algorithm = "global_subset", 
                                    force_continued_learning = FALSE, bootstrap = FALSE, verbose = 0){
  
  gen.loss = matrix(0,Nrounds+1,1)
  train.loss = matrix(0,Nrounds,1)
  nleaves = matrix(0,Nrounds,1)
  train.pred = matrix(numeric(length(y.train)))
  modlist = list()
  
  i = 1
  
  reduction = TRUE
  
  if (samSize == 1 & colsample_bytree == 1 & !bootstrap){
    x.train.samp <- x.train
    y.train.samp <-  y.train
  } else{
    row_smp <- floor(samSize * nrow(x.train))
    row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
    x.train.samp <- x.train[row_ind, ]
    y.train.samp <- y.train[row_ind, ]
    if(colsample_bytree != 1){
      col_smp <- floor(colsample_bytree * ncol(x.train))
      col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
      x.train.samp <- x.train.samp[,col_ind]
    }
  }
  
  #Train first tree with initial prediction
  cand.mod <- gbt.train(y.train.samp, x.train.samp, learning_rate = learnRate, loss_function = loss_function,
                        nrounds = 1, verbose = verbose, greedy_complexities = F, algorithm = algorithm)
  sample_loss <- loss_mse(y.train.samp, predict(cand.mod, x.train.samp))
  full_loss <- loss_mse(y.train, train.pred + predict(cand.mod, x.train))
  gen.loss[1,1] <- cand.mod$estimate_generalization_loss(1) - sample_loss + full_loss
  
  
  while (reduction & i<(Nrounds+1)){
    
    modlist <- c(modlist, list(cand.mod))
    
    nleaves[i,1] <- cand.mod$get_num_leaves()
    
    
    if(type == "reg"){
      if (colsample_bytree == 1){
        train.pred <- train.pred + predict(cand.mod, x.train)
      } else{
        train.pred <- train.pred + predict(cand.mod, x.train[, col_ind])
      }
      train.loss[i,1] = loss_mse(train.pred, y.train)
    } else{
      if(type == "cla"){
        if (colsample_bytree == 1){
          train.pred <- train.pred + logit(predict(camp.mod, x.train))
        } else{
          train.pred <- train.pred + logit(predict(camp.mod, x.train[,col_ind]))
        }
        train.loss[i,1] = LogLossBinary(train.pred, y.train)
      }
    }
    
    if (samSize == 1 & colsample_bytree == 1 & !bootstrap){
      x.train.samp <- x.train
      y.train.samp <-  y.train
      train.pred.samp <- train.pred
    } else{
      row_smp <- floor(samSize * nrow(x.train))
      row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
      x.train.samp <- x.train[row_ind, ]
      y.train.samp <- y.train[row_ind, ]
      train.pred.samp <- train.pred[row_ind, ]
      if(colsample_bytree != 1){
        col_smp <- floor(colsample_bytree * ncol(x.train))
        col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
        x.train.samp <- x.train.samp[,col_ind]
      }
    }
    
    #Train trees without initial prediction by previous_pred
    cand.mod <- gbt.train(y.train.samp, x.train.samp, learning_rate = learnRate, 
                          loss_function = loss_function, nrounds = 1,
                          verbose = verbose, greedy_complexities = F, 
                          previous_pred = train.pred.samp,algorithm = algorithm)
    i = i+1
    sample_loss <- loss_mse(y.train.samp, predict(cand.mod, x.train.samp))
    full_loss <- loss_mse(y.train, train.pred + predict(cand.mod, x.train))
    gen.loss[i,1] <- cand.mod$estimate_generalization_loss(1) - sample_loss + full_loss + gen.loss[(i-1), 1] - train.loss[(i-1), 1]
    
    
    if(!force_continued_learning){
      full.mod <- gbt.train(y.train, x.train, learning_rate = learnRate, 
                            loss_function = loss_function, nrounds = 2,
                            verbose = verbose, previous_pred = train.pred, 
                            greedy_complexities = F, algorithm = algorithm, 
                            force_continued_learning = T)
      reduction = (full.mod$estimate_generalization_loss(1)-full.mod$estimate_generalization_loss(2) > 0)  
    }
    
  }
  
  gen.loss = gen.loss[1:i, 1]
  nleaves = nleaves[1:(i-1), 1]
  train.loss = train.loss[1:(i-1), 1]
  
  
  mod <- list("mod" = modlist, "gen.loss" = gen.loss, "train.loss" = train.loss, 
              "Nleaves" = nleaves, "Nrounds" = Nrounds, "SmpSize" = samSize)
  
  return (mod)
}




# Since the estimated generalization loss for models with sampled training 
# data of size < n is not directly comparable when the samples are different, 
# I propose a new criteria for early
# stopping. For each new tree to an ensamble, I train a new one-tree model on the
# same sample of data with the sample model predictions, and if this is able to
# reduce the estimated generalization relative to the last tree, 
# we keep training.

same.sampling.agt.train <- function(x.train, y.train, learnRate = 0.1, samSize = 1, colsample_bytree = 1, Nrounds = 1000, 
                                    loss_function = "mse", type = "reg", 
                                    algorithm = "global_subset", force_continued_learning = FALSE, 
                                    bootstrap = FALSE, verbose = 0, use_all = F){
  
  gen.loss = numeric(length = Nrounds +1)
  train.loss = numeric(length = Nrounds+1)
  nleaves = numeric(length = Nrounds+1)
  train.pred = matrix(numeric(length(y.train)))
  modlist = list()
  
  i = 1
  
  reduction = TRUE
  
  if (samSize == 1){
    x.train.samp = x.train
    y.train.samp = y.train
  } else{
    row_smp <- floor(samSize * nrow(x.train))
    col_smp <- floor(colsample_bytree * ncol(x.train))
    row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
    col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
    x.train.samp <- x.train[row_ind, col_ind]
    y.train.samp <- y.train[row_ind, ]
  }
  
  #Train first tree with initial prediction
  
  cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, loss_function = loss_function,
                        nrounds = 1, verbose = verbose, greedy_complexities = F, algorithm = algorithm)
  gen.loss[i] <- cand.mod$estimate_generalization_loss(1)
  sample_loss <- loss_mse(y.train.samp, predict(cand.mod, x.train.samp))
  
  
  if(type == "reg"){
    if (colsample_bytree == 1){
      train.pred <- train.pred + predict(cand.mod, x.train)
    } else{
      train.pred <- train.pred + predict(cand.mod, x.train[, col_ind])
    }
    
    train.loss[i] = loss_mse(train.pred, y.train)
    
  }else{
    if(type == "cla"){
      if (colsample_bytree == 1){
        train.pred <- train.pred + logit(predict(camp.mod, x.train))
      } else{
        train.pred <- train.pred + logit(predict(camp.mod, x.train[,col_ind]))
      }
      train.loss[i] = LogLossBinary(train.pred, y.train)
    }
  }
  
  
  while (reduction & i<(Nrounds+1)){
    
    modlist <- c(modlist, list(cand.mod))
    
    nleaves[i] <- cand.mod$get_num_leaves()
    
    i = i + 1
    
    if (samSize == 1 & colsample_bytree == 1){
      x.train.samp <- x.train
      y.train.samp <-  y.train
      train.pred.samp <- train.pred
    } else{
      row_smp <- floor(samSize * nrow(x.train))
      row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
      x.train.samp <- x.train[row_ind, ]
      y.train.samp <- y.train[row_ind, ]
      train.pred.samp <- train.pred[row_ind, ]
      if(colsample_bytree != 1){
        col_smp <- floor(colsample_bytree * ncol(x.train))
        col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
        x.train.samp <- x.train.samp[,col_ind]
      }
    }
    
    
    cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, 
                          loss_function = loss_function, nrounds = 1,
                          verbose = verbose, greedy_complexities = F, 
                          previous_pred = train.pred.samp,algorithm = algorithm)
    
    
    gen.loss[i] <-  cand.mod$estimate_generalization_loss(1) + gen.loss[(i-1)] - sample_loss
    sample_loss <- loss_mse(y.train.samp, train.pred.samp + predict(cand.mod, x.train.samp))
    
    
    if(type == "reg"){
      if (colsample_bytree == 1){
        train.pred <- train.pred + predict(cand.mod, x.train)
      } else{
        train.pred <- train.pred + predict(cand.mod, x.train[, col_ind])
      }
      train.loss[i] = loss_mse(train.pred, y.train)
    } else{
      if(type == "cla"){
        if (colsample_bytree == 1){
          train.pred <- train.pred + logit(predict(camp.mod, x.train))
        } else{
          train.pred <- train.pred + logit(predict(camp.mod, x.train[,col_ind]))
        }
        train.loss[i] = LogLossBinary(train.pred, y.train)
      }
    }
    
    
    if (samSize == 1 && colsample_bytree == 1){
      train.pred.samp <- train.pred
    } else{
      train.pred.samp <- train.pred[row_ind, ]
      if(colsample_bytree != 1){
        train.pred.samp <- train.pred.samp[,col_ind]
      }
    }
    
    
    cand2.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, 
                           loss_function = loss_function, nrounds = 1,
                           verbose = 0, greedy_complexities = F, 
                           previous_pred = train.pred.samp,algorithm = algorithm)
    
    cand2.gen.loss <- cand2.mod$estimate_generalization_loss(1) + gen.loss[i] - sample_loss
    
    
    if(!force_continued_learning){
      reduction = !(gen.loss[i] - cand2.gen.loss < 0)  # Need to find this expression
    }
    
  }
  
  gen.loss = gen.loss[1:(i+1)]
  nleaves = nleaves[1:(i-1)]
  train.loss = train.loss[1:(i-1)]
  
  
  mod <- list("mod" = modlist, "gen.loss" = gen.loss, "train.loss" = train.loss, 
              "Nleaves" = nleaves, "Nrounds" = Nrounds, "SmpSize" = samSize, "cand2.gen.loss" = cand2.gen.loss)
  
  return (mod)
}


#Function for predicting with sampling.agt model. 
#Loops through all trees and adds to prediction.
sampling.agt.pred <- function(mod, Xtest, type = "reg"){
  mod = mod$mod
  test.pred = matrix(numeric(nrow(y.test)))
  if(type == "reg"){
    for (i in 1:length(mod)){
      test.pred <- test.pred + predict(mod[[i]], Xtest)
    }
  }else{
    if(type == "cla"){
      for (i in 1:length(mod)){
        test.pred <- test.pred + logit(predict(mod[[i]], Xtest))
      }
      test.pred <- logistic(test.pred)
    }
  }
  return(test.pred)
}


#Function for calculating loss with sampling.agt model.
#Equal to sampling.agt.pred, but calculates loss at each iteration
sampling.agt.loss <- function(mod, Xtest, Ytest, type = "reg"){
  mod = mod$mod
  len = length(mod)
  loss = matrix(numeric(len))
  test.pred = matrix(numeric(length(Ytest)))
  if(type == "reg"){
    for (i in 1:len){
      test.pred <- test.pred + predict(mod[[i]], Xtest)
      loss[i] <- loss_mse(Ytest, test.pred)
    }
  }else{
    if(type == "cla"){
      for (i in 1:len){
        test.pred <- test.pred + logit(predict(mod[[i]], Xtest))
        loss[i] <- LogLossBinary(Ytest, logistic(test.pred))
      }
    }
  }
  
  return(loss)
}

#Function for calculating loss for agt model. 
agt.loss <- function(mod, Xtest, Ytest, type = "reg"){
  loss = matrix(numeric(length(mod$get_num_trees())))
  if(type == "reg"){
    for (i in 1:mod$get_num_trees()){
      test.pred <- mod$predict2(Xtest, i)
      loss[i] <- loss_mse(Ytest, test.pred)
    }
  }else{
    if(type == "cla"){
      for (i in 1:mod$get_num_trees()){
        test.pred <- mod$predict2(Xtest, i)
        loss[i] <- loss_mse(Ytest, logistic(test.pred))
      }
    }
  }
  
  return(loss)
}





# Playing with sample sizes
#First off, i propose to use a progressive sampling size. As the sample size decreases,
#the chance of random effects due to sample size outweights the systematic effects inherent
#in the data increases. So, what if we start off by training with a low sampling size,
#and then reduce it when training on this sampling size is no longer helping the 
#estimated generalization loss. Since we with lower sampling size must expect higher 
#variance in the estimated generalization loss, we allow for a one-step increase in 
#estimated generalization loss no bigger than a/s, where s is the current sampling size
#and a is some parameter >= 0 


step.sampling.agt.train <- function(x.train, y.train, learnRate, samSize, colsample_bytree = 1, Nrounds, loss_function = "mse", 
                                    type = "reg", algorithm = "global_subset", force_continued_learning = FALSE, 
                                    bootstrap = FALSE, verbose = 0, A = 0, stepSize = 0.1){
  
  gen.loss = matrix(0,Nrounds+1,1)
  train.loss = matrix(0,Nrounds,1)
  sam.sizes = numeric(Nrounds + 1)
  nleaves = matrix(0,Nrounds,1)
  train.pred = matrix(numeric(length(y.train)))
  modlist = list()
  
  i = 1
  
  
  if (samSize == 1 & colsample_bytree == 1){
    x.train.samp <- x.train
    y.train.samp <-  y.train
  } else{
    row_smp <- floor(samSize * nrow(x.train))
    row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
    x.train.samp <- x.train[row_ind, ]
    y.train.samp <- y.train[row_ind, ]
    if(colsample_bytree != 1){
      col_smp <- floor(colsample_bytree * ncol(x.train))
      col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
      x.train.samp <- x.train.samp[,col_ind]
    }
  }
  
  #Train first tree with initial prediction
  cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, loss_function = loss_function,
                        nrounds = 1, verbose = verbose, greedy_complexities = F, algorithm = algorithm)
  gen.loss[1,1] <- cand.mod$estimate_generalization_loss(1)
  
  
  step_reduction = TRUE
  
  while(step_reduction & i<(Nrounds+1)){
    reduction = TRUE
    
    while (reduction & i<(Nrounds+1)){
      
      modlist <- c(modlist, list(cand.mod))
      
      nleaves[i,1] <- cand.mod$get_num_leaves()
      sam.sizes[i] <- samSize
      
      if(type == "reg"){
        if (colsample_bytree == 1){
          train.pred <- train.pred + predict(cand.mod, x.train)
        } else{
          train.pred <- train.pred + predict(cand.mod, x.train[, col_ind])
        }
        train.loss[i,1] = loss_mse(train.pred, y.train)
      } else{
        if(type == "cla"){
          if (colsample_bytree == 1){
            train.pred <- train.pred + logit(predict(camp.mod, x.train))
          } else{
            train.pred <- train.pred + logit(predict(camp.mod, x.train[,col_ind]))
          }
          train.loss[i,1] = LogLossBinary(train.pred, y.train)
        }
      }
      
      if (samSize == 1 & colsample_bytree == 1){
        x.train.samp <- x.train
        y.train.samp <-  y.train
        train.pred.samp <- train.pred
      } else{
        row_smp <- floor(samSize * nrow(x.train))
        row_ind <- sample(seq_len(nrow(x.train)), size = row_smp, replace = bootstrap)
        x.train.samp <- x.train[row_ind, ]
        y.train.samp <- y.train[row_ind, ]
        train.pred.samp <- train.pred[row_ind, ]
        if(colsample_bytree != 1){
          col_smp <- floor(colsample_bytree * ncol(x.train))
          col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
          x.train.samp <- x.train.samp[,col_ind]
        }
      }
      
      #Train trees without initial prediction by previous_pred
      cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, 
                            loss_function = loss_function, nrounds = 1,
                            verbose = verbose, greedy_complexities = F, 
                            previous_pred = train.pred.samp,algorithm = algorithm)
      i = i+1
      
      gen.loss[i,1] <- cand.mod$estimate_generalization_loss(1) + gen.loss[(i-1), 1] - train.loss[(i-1), 1]
      
      if(!force_continued_learning & i>5){
        reduction = (gen.loss[(i-1),1]/gen.loss[(i),1] > samSize && 
                       gen.loss[i,1] - gen.loss[(i-5),1] < 0)  # Need to find this expression
      }
      
    }
    
    
    if(samSize == 1){
      step_reduction = FALSE
    } else{
      samSize = samSize + stepSize
    }
    
    if(samSize > 1){
      samSize = 1
    }
  }
  
  gen.loss = gen.loss[1:i, 1]
  nleaves = nleaves[1:(i-1), 1]
  train.loss = train.loss[1:(i-1), 1]
  sam.sizes = sam.sizes[1:i]
  
  
  mod <- list("mod" = modlist, "gen.loss" = gen.loss, "train.loss" = train.loss, 
              "Nleaves" = nleaves, "Nrounds" = Nrounds, "SamSizes" = sam.sizes)
  
  return (mod)
}





#Trying with GOSS sampling as described in LightGBM, where we sample the 
#a residuals with highest gradients, and b times the (1-a) lowest residuals.
#Can be combined with any early stopping method

goss.sampling.agt.train <- function(x.train, y.train, learnRate = 0.1, samSize = 1, colsample_bytree = 1, 
                                    Nrounds = 1000,  loss_function = "mse", 
                                    type = "reg", algorithm = "global_subset", force_continued_learning = FALSE, 
                                    bootstrap = FALSE, verbose = 0, a = 0.1, b = samSize/(1-a)){
  
  gen.loss = matrix(0,Nrounds+1,1)
  train.loss = matrix(0,Nrounds,1)
  sam.sizes = numeric(Nrounds + 1)
  nleaves = matrix(0,Nrounds,1)
  train.pred = matrix(numeric(length(y.train)))
  modlist = list()
  
  i = 1
  
  
  #Train first tree with initial prediction
  cand.mod <- gbt.train(y.train, x.train, learnRate, loss_function = loss_function,
                        nrounds = 1, verbose = verbose, greedy_complexities = F, algorithm = algorithm)
  gen.loss[1,1] <- cand.mod$estimate_generalization_loss(1)
  
  
  reduction = T
  
  while (reduction & i<(Nrounds+1)){
    
    modlist <- c(modlist, list(cand.mod))
    
    nleaves[i,1] <- cand.mod$get_num_leaves()
    
    if(type == "reg"){
      if (colsample_bytree == 1){
        train.pred <- train.pred + predict(cand.mod, x.train)
      } else{
        train.pred <- train.pred + predict(cand.mod, x.train[, col_ind])
      }
      train.loss[i,1] = loss_mse(train.pred, y.train)
      grads = order(abs(train.pred - y.train), decreasing = TRUE)
      
    } else{
      if(type == "cla"){
        if (colsample_bytree == 1){
          train.pred <- train.pred + logit(predict(camp.mod, x.train))
        } else{
          train.pred <- train.pred + logit(predict(camp.mod, x.train[,col_ind]))
        }
        train.loss[i,1] = LogLossBinary(train.pred, y.train)
      }
    }
    
    
    top_smp <- floor(a * length(grads))
    
    btm_smp <- floor(b * (1-a) * length(grads))
    
    row_ind <- grads[1:top_smp]
    btm_ind <- sample(grads[-1:-top_smp], size = btm_smp, replace = FALSE)
    
    row_ind = c(row_ind, btm_ind)
    
    x.train.samp <- x.train[row_ind, ]
    y.train.samp <- y.train[row_ind, ]
    train.pred.samp <- train.pred[row_ind, ]
    
    
    if(colsample_bytree != 1){
      col_smp <- floor(colsample_bytree * ncol(x.train))
      col_ind <- sample(seq_len(ncol(x.train)), size = col_smp, replace = FALSE)
      x.train.samp <- x.train.samp[,col_ind]
    }
    
    last_train_loss = loss_mse(y.train.samp, predict(cand.mod, x.train.samp))
    
    #Train trees without initial prediction by previous_pred
    cand.mod <- gbt.train(y.train.samp, x.train.samp, learnRate, 
                          loss_function = loss_function, nrounds = 1,
                          verbose = verbose, greedy_complexities = F, 
                          previous_pred = train.pred.samp,algorithm = algorithm)
    i = i+1
    
    gen.loss[i,1] <- cand.mod$estimate_generalization_loss(1) + gen.loss[(i-1), 1] - train.loss[(i-1), 1]
    
    this_train_loss = loss_mse(y.train.samp, predict(cand.mod, x.train.samp))
    
    if(!force_continued_learning & i>2){
      reduction = (gen.loss[i,1] - gen.loss[(i-1),1] < 0)  # Need to find this expression
    }
    
  }
  
  
  gen.loss = gen.loss[1:i, 1]
  nleaves = nleaves[1:(i-1), 1]
  train.loss = train.loss[1:(i-1), 1]
  
  
  mod <- list("mod" = modlist, "gen.loss" = gen.loss, "train.loss" = train.loss, 
              "Nleaves" = nleaves, "Nrounds" = Nrounds)
  
  return (mod)
}




#Dataset generator function
dataset <- function(i, seed){
  
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
    x.train <<- as.matrix(Boston[ind_train,-medv_col])
    y.train <<- as.matrix(log(Boston[ind_train,medv_col]))
    x.test <<- as.matrix(Boston[-ind_train,-medv_col])
    y.test <<- as.matrix(log(Boston[-ind_train,medv_col]))
    
  }else 
    if(i==2){
      # ozone
      data(ozone)
      n_full <- nrow(ozone)
      ind_train <- sample(n_full, 0.5*n_full)
      x.train <<- as.matrix(log(ozone[ind_train,-1]))
      y.train <<- as.matrix(log(ozone[ind_train,1]))
      x.test <<- as.matrix(log(ozone[-ind_train,-1]))
      y.test <<- as.matrix(log(ozone[-ind_train,1]))
      
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
                          
                        }
} 


