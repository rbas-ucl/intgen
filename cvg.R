# Load libraries, register cores
library(data.table)
library(doMC)
registerDoMC(8)

# Set seed
set.seed(123, kind = "L'Ecuyer-CMRG")

### IFM coverage experiment ###

# Hyperparameters
n_per_regime <- 5000
alpha <- 0.1
regimes <- seq(-2, 2, by = 0.5)
trn_regimes <- -2:2
tst_regimes <- setdiff(regimes, trn_regimes)

# Optionally weighted quantile function (from Tibshirani et al., 2019)
weighted_quantile = function(v, prob, w = NULL, sorted = FALSE) {
  if (is.null(w)) {
    w <- rep(1, length(v))
  } 
  if (!sorted) { 
    o <- order(v)
    v <- v[o]
    w <- w[o] 
  }
  i <- which(cumsum(w / sum(w)) >= prob)
  if (length(i) == 0) {
    out <- Inf 
  } else {
    out <- v[min(i)]
  }
  return(out)
}

# Mu function is basically a kernel regression, with weights proportional to
# the true likelihood ratio (oracle access)
mu_fn <- function(df, target_sigma) {
  tmp <- df[regime != target_sigma]
  lr <- tmp[, dnorm(x, mean = target_sigma) / dnorm(x, mean = regime)]
  w <- lr / sum(lr)
  #f <- lm(y ~ I(x^2) + x, tmp)
  f <- lm(y ~ I(x^2), tmp)
  out <- as.numeric(crossprod(w, fitted(f)))
  return(out)
}

# Simulation loop
loop <- function(b, weighted = TRUE) {
  
  # Populate
  dat <- data.table(regime = rep(seq(-2, 2, by = 0.5), each = n_per_regime))
  n <- dat[, .N]
  dat[, x := rnorm(n, mean = regime)]
  #dat[, y := x^2 + x + rnorm(n, sd = 0.5)]
  dat[, y := x^2 + rnorm(n, sd = x^2)]
  dat[, set := fifelse(regime %in% trn_regimes, 'trn', 'tst')]
  n_trn <- dat[set == 'trn', .N]
  
  # Fit models, compute conformity scores
  mu_dat <- data.table(regime = regimes)
  mu_dat[, mu_hat := sapply(regimes, function(k) mu_fn(dat, k))]
  dat <- merge(dat, mu_dat, by = 'regime', all.x = TRUE)
  dat[set == 'trn', s := abs(y - mu_hat)]
  
  # Compute taus
  tau_fn <- function(target_sigma, weighted) {
    if (isTRUE(weighted)) {
      lr <- dat[set == 'trn', dnorm(x, mean = target_sigma) / dnorm(x, mean = regime)]
      tau <- weighted_quantile(dat[set == 'trn', s], 1 - alpha, w = lr)
    } else {
      tau <- weighted_quantile(dat[set == 'trn', s], 1 - alpha)
    }
    return(tau)
  }
  tau_dat <- data.table(regime = tst_regimes)
  tau_dat[, tau_hat := sapply(tst_regimes, function(k) tau_fn(k, weighted))]
  dat <- merge(dat, tau_dat, by = 'regime', all.x = TRUE)
  
  # Success?
  dat[, covered := fifelse((y >= mu_hat - tau_hat) & (y <= mu_hat + tau_hat), 1, 0)]
  rate <- dat[set == 'tst', sum(covered) / .N]
  return(rate)
  
}
rates <- foreach(bb = 1:1000, .combine = c) %dopar% loop(bb, TRUE)


