# Load libraries, register cores
library(data.table)
library(doMC)
registerDoMC(8)

# Set seed
set.seed(123, type = "L'Ecuyer-CMRG")

### IFM coverage experiment ###

# Hyperparameters
n_per_regime <- 5000
alpha <- 0.1
regimes <- seq(-2, 2, by = 0.5)
trn_regimes <- -2:2
tst_regimes <- setdiff(regimes, trn_regimes)

# Mu function is basically a kernel regression, with weights proportional to
# the true likelihood ratio (oracle access)
mu <- function(df, target_sigma) {
  tmp <- df[regime != target_sigma]
  lr <- tmp[, dnorm(x, mean = target_sigma) / dnorm(x, mean = regime)]
  w <- lr / sum(lr)
  f <- lm(y ~ I(x^2) + x, tmp)
  out <- as.numeric(crossprod(w, fitted(f)))
  return(out)
}

# Repeat a bunch
loop <- function(b) {
  
  # Populate
  dat <- data.table(regime = rep(seq(-2, 2, by = 0.5), each = n_per_regime))
  n <- dat[, .N]
  dat[, x := rnorm(n, mean = regime)]
  dat[, y := x^2 + x + rnorm(n, sd = 0.5)]
  dat[, set := fifelse(regime %in% trn_regimes, 'trn', 'tst')]
  n_trn <- dat[set == 'trn', .N]
  
  # Fit models, compute conformity scores
  mu_dat <- data.table(regime = regimes)
  mu_dat[, mu_hat := sapply(regimes, function(k) mu(dat, k))]
  dat <- merge(dat, mu_dat, by = 'regime', all.x = TRUE)
  dat[set == 'trn', s := abs(y - mu_hat)]
  q <- ceiling((n_trn + 1) * (1 - alpha))
  tau <- sort(dat[set == 'trn', s])[q] # This should hold marginally across all regimes
  
  # Success?
  dat[, covered := fifelse((y > mu_hat - tau) & (y < mu_hat + tau), 1, 0)]
  rate <- dat[set == 'tst', sum(covered) / .N]
  return(rate)
  
}
rates <- foreach(bb = 1:1000, .combine = c) %dopar% loop(bb)



