



library(MASS)
library(ISLR)
library(ElemStatLearn)
library(agtboost)




dataset(8,123)
set.seed(123)

gen.loss = numeric(1000)
tr.loss = numeric(1000)
test.loss = numeric(1000)

sampling_size = 0.2

row_smp <- floor(sampling_size * nrow(x.train))
row_ind <- sample(seq_len(nrow(x.train)), size = row_smp)
x.train.samp <- x.train[row_ind, ]
y.train.samp <- y.train[row_ind, ]


set.seed(123)
mod1 = gbt.train(y.train.samp, x.train.samp, learning_rate = 0.1, nrounds = 1, verbose = 1, type = "cla", loss_function = "logloss")
preds1 = logit(predict(mod1, x.train))
preds2 = logit(predict(mod1, x.train.samp))
test.preds = logit(predict(mod1, x.test))
tr.loss[1] = LogLossBinary(y.train, logistic(preds1))
samp.loss = LogLossBinary(y.train.samp, logistic(preds2))
opti = sampling_size * (mod1$estimate_generalization_loss(1) - samp.loss)
gen.loss[1] =  tr.loss[1] +  opti

test.loss[1] = LogLossBinary(y.test, logistic(test.preds))

continue=T
i = 2
min_loss = Inf
while(continue & i<113){
  row_smp <- floor(sampling_size * nrow(x.train))
  row_ind <- sample(seq_len(nrow(x.train)), size = row_smp)
  x.train.samp <- x.train[row_ind, ]
  y.train.samp <- y.train[row_ind, ]
  pred.samp <- preds1[row_ind]
  y.train.samp2 <- y.train2[row_ind, ]
  
  mod2 = gbt.train(y.train.samp, x.train.samp, learning_rate = 0.1, previous_pred = pred.samp,
                   nrounds = 1, verbose = 1, loss_function = "logloss")
  
  preds1 = pmin(pmax(preds1 + logit(predict(mod2, x.train)), -32.23619), 32.23699)
  preds2 = logit(predict(mod1, x.train.samp))
  test.preds = test.preds + logit(predict(mod2, x.test))
  samp.loss = LogLossBinary(y.train.samp, logistic(preds1[row_ind]))
  tr.loss[i] = LogLossBinary(y.train, logistic(preds1))
  opti = opti + sampling_size * (mod2$estimate_generalization_loss(1) - samp.loss)
  gen.loss[i] = tr.loss[i] + opti
  test.loss[i] = LogLossBinary(y.test, logistic(test.preds))
  if(gen.loss[i] < min_loss){
    min_loss = gen.loss[i]
    min = i
  }
  if(i-200 > min){
    continue = F
  }
  i = i+1
  cat("minimum prediction: ", min(preds1), "\n")
  cat("minimum prediction contribution from last tree: ", min(logit(predict(mod2, x.train))), "\n")
}

gen.loss = gen.loss[1:(min+201)]
tr.loss = tr.loss[1:(min+201)]
test.loss = test.loss[1:(min+201)]

plot(tr.loss, ylim = c(0,0.3))
points(gen.loss, col= "blue")
points(test.loss, col ="orange")
abline(v=min, col = "red")
