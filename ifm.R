# Load libraries
library(data.table)
library(foreach)
library(ggplot2)
library(ggsci)

# Upfront
BENCHMARK_PATH =  "~/current_experiments/"

# Data prep function
data_prep_fn <- function(dat, sim) {
  BENCHMARK_CHOICE <- paste0(dat, '_', sim, 'truth_results')
  SIGMA_CHOICE_ALL <- paste0(dat, '_sigma_all')
  
  sigma_all = as.matrix(read.csv(paste(BENCHMARK_PATH, SIGMA_CHOICE_ALL, ".csv", sep=""), header = FALSE))
  true_y = as.matrix(read.csv(paste(BENCHMARK_PATH, BENCHMARK_CHOICE, "_true.csv", sep=""), header = FALSE))
  y_hat_blackbox = as.matrix(read.csv(paste(BENCHMARK_PATH, BENCHMARK_CHOICE, "_blackbox.csv", sep=""), header = FALSE))
  y_hat_dag = as.matrix(read.csv(paste(BENCHMARK_PATH, BENCHMARK_CHOICE, "_y_hat_dag.csv", sep=""), header = FALSE))
  y_hat_ifm_1 = as.matrix(read.csv(paste(BENCHMARK_PATH, BENCHMARK_CHOICE, "_y_hat_ifm_1_mlp.csv", sep=""), header = FALSE))
  y_hat_ifm_2 = as.matrix(read.csv(paste(BENCHMARK_PATH, BENCHMARK_CHOICE, "_y_hat_ifm_3_mlp.csv", sep=""), header = FALSE))
  y_hat_ifm_3 = as.matrix(read.csv(paste(BENCHMARK_PATH, BENCHMARK_CHOICE, "_y_hat_ifm_4_mlp.csv", sep=""), header = FALSE))
  true_var_y = as.matrix(read.csv(paste(BENCHMARK_PATH, BENCHMARK_CHOICE, "_true_var_y.csv", sep=""), header = FALSE))
  
  num_problems = 100
  
  ######### SUMMARY STATISTICS
  
  
  # Now, calculate summary statistics. This is done by taking, for each of the 100 outcome problems, 
  # a measure of comparison between the ground truth $d$-dimensional vector against the estimated 
  # $d$-dimensional vector. We drop all training set rgimes though, `include_sigma` below refers to 
  # unseen regimes.
  #
  # `rmse` is root mean squared error. Not too meaningful without knowing the scale of the variable. So we 
  # have also `pmse`, which is the MSE divided by the variance of the outcome variable in each problem.
  #
  # The other main measure is rank correlation (Spearman's $\rho$). If
  
  rmse_dag = rep(0, num_problems)
  spearman_dag = rep(0, num_problems)
  prmse_dag = rep(0, num_problems)
  
  rmse_ifm1 = rep(0, num_problems)
  spearman_ifm1 = rep(0, num_problems)
  prmse_ifm1 = rep(0, num_problems)
  
  rmse_ifm2 = rep(0, num_problems)
  spearman_ifm2 = rep(0, num_problems)
  prmse_ifm2 = rep(0, num_problems)
  
  rmse_ifm3 = rep(0, num_problems)
  spearman_ifm3 = rep(0, num_problems)
  prmse_ifm3 = rep(0, num_problems)
  
  rmse_blackbox = rep(0, num_problems)
  spearman_blackbox = rep(0, num_problems)
  prmse_blackbox = rep(0, num_problems)
  
  include_sigma = which(rowSums(sigma_all) > 1) # Only include regimes with two or more non-zero actions
  
  for (p in 1:num_problems) {
    
    rmse_dag[p] = sqrt(mean((true_y[p, include_sigma] - y_hat_dag[p, include_sigma])^2))
    rmse_ifm1[p] = sqrt(mean((true_y[p, include_sigma] - y_hat_ifm_1[p, include_sigma])^2))
    rmse_ifm2[p] = sqrt(mean((true_y[p, include_sigma] - y_hat_ifm_2[p, include_sigma])^2))
    rmse_ifm3[p] = sqrt(mean((true_y[p, include_sigma] - y_hat_ifm_3[p, include_sigma])^2))
    rmse_blackbox[p] = sqrt(mean((true_y[p, include_sigma] - y_hat_blackbox[p, include_sigma])^2))
    
    spearman_dag[p] = cor(true_y[p, include_sigma], y_hat_dag[p, include_sigma], method = "spearman")
    spearman_ifm1[p] = cor(true_y[p, include_sigma], y_hat_ifm_1[p, include_sigma], method = "spearman")
    spearman_ifm2[p] = cor(true_y[p, include_sigma], y_hat_ifm_2[p, include_sigma], method = "spearman")
    spearman_ifm3[p] = cor(true_y[p, include_sigma], y_hat_ifm_3[p, include_sigma], method = "spearman")
    spearman_blackbox[p] = cor(true_y[p, include_sigma], y_hat_blackbox[p, include_sigma], method = "spearman")
    
    prmse_dag[p] = mean((true_y[p, include_sigma] - y_hat_dag[p, include_sigma])^2 / true_var_y[p, include_sigma])
    prmse_ifm1[p] = mean((true_y[p, include_sigma] - y_hat_ifm_1[p, include_sigma])^2 / true_var_y[p, include_sigma])
    prmse_ifm2[p] = mean((true_y[p, include_sigma] - y_hat_ifm_2[p, include_sigma])^2 / true_var_y[p, include_sigma])
    prmse_ifm3[p] = mean((true_y[p, include_sigma] - y_hat_ifm_3[p, include_sigma])^2 / true_var_y[p, include_sigma])
    prmse_blackbox[p] = mean((true_y[p, include_sigma] - y_hat_blackbox[p, include_sigma])^2 / true_var_y[p, include_sigma])
    
  }
  
  # Export
  out <- data.table(
    'Data' = dat, 'Simulation' = sim, 
    'Model' = rep(c('Black Box', 'DAG', 'IFM1', 'IFM2', 'IFM3'), each = 100),
    'pRMSE' = c(prmse_blackbox, prmse_dag, prmse_ifm1, prmse_ifm2, prmse_ifm3),
    'rCOR' = c(spearman_blackbox, spearman_dag, spearman_ifm1, spearman_ifm2, spearman_ifm3)
  ) 
  return(out)
}
df <- foreach(dd = c('dream', 'sachs'), .combine = rbind) %:%
  foreach(ss = c('dag', 'ifm'), .combine = rbind) %do% data_prep_fn(dd, ss)
df[, Data := ifelse(Data == 'dream', 'DREAM', 'Sachs')]
df[, Simulation := ifelse(Simulation == 'dag', 'DAG', 'IFM')]

# Boxplots
box_fn <- function(data, sim) {
  idx <- grepl(data, df$Data) & grepl(sim, df$Simulation)
  tmp <- df[idx, ]
  tmp[, Data := paste0('Dataset: ', Data)]
  tmp[, Simulation := paste0('Simulation: ', Simulation)]
  p <- ggplot(tmp, aes(Model, pRMSE, fill = Model)) + 
    geom_boxplot() + 
    facet_wrap(~ Data + Simulation, nrow = 1) + 
    scale_fill_lancet() + 
    labs(x = '', y = '') +
    theme_bw() +
    theme(legend.position = 'none', text = element_text(size = 25))
  ggsave(paste0(data, '_', sim, '_box.pdf'), width = 6, height = 6)
}
foreach(dd = c('DREAM', 'Sachs')) %:%
  foreach(ss = c('DAG', 'IFM')) %do% box_fn(dd, ss)


# Histograms
df[, Data := paste0('Dataset: ', Data)]
df[, Simulation := paste0('Simulation: ', Simulation)]
ggplot(df[Model %in% c('Black Box', 'IFM1')], aes(x = rCOR, fill = Model)) +
  geom_density(alpha = 0.4, color = NA) + 
  scale_fill_lancet() + 
  labs(x = 'Rank Correlation', y = 'Density') +
  facet_wrap(~ Data + Simulation, scales = 'free', nrow = 1) +
  theme_bw() + 
  theme(text = element_text(size = 25))



# One-sided binomial tests
test_fn <- function(dat, sim, mod) {
  e0 <- df[data == dat & simulation == sim & model == 'Black Box', error]
  e1 <- df[data == dat & simulation == sim & model == mod, error]
  delta <- e0 - e1
  res <- binom.test(sum(delta > 0), length(delta), alternative = 'greater')
  out <- data.table(
    'data' = dat, 'simulation' = sim, 'model' = mod, 'p.value' = res$p.value
  )
  return(out)
}
df_tst <- foreach(dd = c('dream', 'sachs'), .combine = rbind) %:%
  foreach(ss = c('dag', 'ifm'), .combine = rbind) %:%
  foreach(mm = c('DAG', 'IFM1', 'IFM2'), .combine = rbind) %do%
  test_fn(dd, ss, mm)
df_tst <- rbind(
  df_tst, foreach(ss = c('dag', 'ifm'), .combine = rbind) %do% 
    test_fn('sachs', ss, 'IFM3')
  )
setkey(df_tst, c('data', 'simulation', 'model'))
p_out <- dcast(df_tst, data + simulation ~ model)



















