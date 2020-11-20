# Lib
library(agtboost)

# Data
n <- 1000
xtr <- as.matrix(runif(n, -5, 5))
ytr <- rnorm(n, xtr^2, 1)
xte <- as.matrix(runif(n, -5, 5))
yte <- rnorm(n, xte^2, 1)
plot(xtr, ytr)

# mse function
mse <- function(y, y.hat) mean((y-y.hat)^2)

# 10 trees
mod1 <- gbt.train(ytr, xtr, verbose=1, learning_rate = 0.01, algorithm = "vanilla",
                  nrounds=10, force_continued_learning = T)
pred1 <- predict(mod1, xtr)

# Train 1 tree more
mod2 <- gbt.train(ytr, xtr, verbose=1, learning_rate = 0.01, algorithm = "vanilla",
                  nrounds=1, force_continued_learning = T, previous_pred = pred1)
pred2 <- predict(mod2, xtr)

# Find training loss
tr_loss_prev <- mse(ytr, pred1)
tr_loss_now <- mse(ytr, pred1+pred2)

# Find optimism: Gen-loss - train-loss
opt_1 <- mod1$estimate_generalization_loss(10) - tr_loss_prev
opt_2 <- mod2$estimate_generalization_loss(1) - tr_loss_now

# Optimism for mod1 + mod2
approx_E_te_loss_now <- tr_loss_now + opt_1 + opt_2

# should approximately equal:
mod3 <- gbt.train(ytr, xtr, verbose=1, learning_rate = 0.01, algorithm = "vanilla",
                  nrounds=11, force_continued_learning = T)
mod3$estimate_generalization_loss(11) # <--- this
approx_E_te_loss_now # But discrapancies due to new CIR. Create issue on set.seed for CIR.
