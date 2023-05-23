#########################################################################
# General tools for benchmarking

using DelimitedFiles, DataFrames
using Plots, StatsPlots
using JLD2, FileIO, CSV

include("other_learning.jl")

########################################################################
# METHODS FOR ESTIMATING EXPECTED OUTCOMES FOR FITTED DAG MODELS OR IFMS

""" `batch_mediator_predict(x_modelhat_montecarlo::Vector{Matrix{Float64}}, sigma::Matrix{Int},
                            x_trains::Vector, y_trains::Vector, verbose::Bool = false)`

Given the regimes described in the rows of `sigma`, and a corresponding Monte Carlo approximation `x_modelhat_montecarlo`
for the corresponding mediator model at that regime, learn a predictor first mapping `x_trains` to `y_trains`, then
marginalize over the (Monte Carlo representation of the) mediator model for any given test regime in `sigma`. 
It then returns the estimated expected outcome for each one of the regimes encoded in `sigma`.
"""
function batch_mediator_predict(x_modelhat_montecarlo::Vector{Matrix{Float64}}, sigma::Matrix{Int},
                                x_trains::Vector, y_trains::Vector, verbose::Bool = false)

    
    if verbose
        println("Making predictions from mediator model...")
    end

    num_problems = length(y_trains)
    num_regimes = size(sigma, 1)
    y_hat = zeros(num_problems, num_regimes)

    verbose ? pb = ProgressBar(1:num_problems) : pb = 1:num_problems

    for p = pb

        x = copy(x_trains[p][1])
        y = copy(y_trains[p][1])
        for i = 2:length(x_trains[p])
            x = vcat(x, x_trains[p][i])
            y = vcat(y, y_trains[p][i])
        end

        y_theta_hat = supervised_train(x, y[:, 1], 5, false)
        for i = 1:num_regimes
            y_hat[p, i] = mean(supervised_prediction(x_modelhat_montecarlo[i], y_theta_hat[1]))
        end

    end

    return y_hat

end

""" `batch_mediator_predict(sigma::Matrix{Int}, x_train::Vector, y_trains::Vector, sigma_train::Vector, ifm::Ifm, 
                           theta::Vector, verbose::Bool = false)`

This version uses no simulations or training whatsover: just the original training data `x_train` with the 
IFM `ifm` under estimated parameters `theta`, with reweighted outcomes. That is, it implements the importance
weighted version of IFM, with no final regression model mapping mediators to outcomes. 
"""
function batch_mediator_predict(sigma::Matrix{Int}, x_train::Vector, y_trains::Vector, sigma_train::Matrix, ifm::Ifm, 
                                theta::Vector, verbose::Bool = false)

    
    num_problems = length(y_trains)
    num_train_regimes, num_test_regimes = size(sigma_train, 1), size(sigma, 1)
    y_hat = zeros(num_problems, num_test_regimes)
    nnets = Matrix(undef, num_problems, num_test_regimes)

    # Get model-based importance weights
    weights = Matrix(undef, num_train_regimes, num_test_regimes)
    for i = 1:num_train_regimes
        for j = 1:num_test_regimes
            weights[i, j] = report_density_ratio_ifm(x_train[i], sigma_train[i, :], sigma[j, :], theta, ifm)
        end
    end

    verbose ? pb = ProgressBar(1:num_problems) : pb = 1:num_problems
    for p = pb
        for j = 1:num_test_regimes
            y_hat_here, y_hat_var_here = zeros(num_train_regimes), zeros(num_train_regimes)
            for i = 1:num_train_regimes
                y_hat_here[i] = sum(y_trains[p][i][:, 1] .* weights[i, j])
                y_hat_var_here[i] = var(y_trains[p][i][:, 1]) * sum(weights[i, j].^2)
            end
            y_hat_weight = (1 ./ y_hat_var_here) ./ sum(1 ./ y_hat_var_here)
            y_hat[p, j] = sum(y_hat_here .* y_hat_weight)
        end
    end

    return y_hat, weights

end

""" `batch_mediator_predict(x_modelhat_montecarlo::Vector{Matrix{Float64}}, sigma::Matrix{Int},
                            x_train::Vector, y_trains::Vector, sigma_train::Matrix, ifm::Ifm, 
                            theta::Vector, num_hidden_space::Vector{Int}, num_folds::Int,
                            rho_nn::Float64, max_iter_nn::Int, verbose::Bool = false)`

This variation implements the (very slow) covariate shift regression approach for the IFM. Here, for
*each* regime in `sigma`, it fits a separate importance-reweighted regression model. This regression
model is a MLP, with hidden units picked from a space of candidates `num_hidden_space` and selected by
`num_folds` cross-validation. Parameters `rho_nn` and `max_iter_nn` control the Adam optimizer used.
"""
function batch_mediator_predict(x_modelhat_montecarlo::Vector{Matrix{Float64}}, sigma::Matrix{Int},
                                x_train::Vector, y_trains::Vector, sigma_train::Matrix, ifm::Ifm, 
                                theta::Vector, num_hidden_space::Vector{Int}, num_folds::Int,
                                rho_nn::Float64, max_iter_nn::Int, verbose::Bool = false)

    
    if verbose
        println("Making predictions from mediator model...")
    end

    num_problems = length(y_trains)
    num_train_regimes, num_test_regimes = size(sigma_train, 1), size(sigma, 1)
    y_hat = zeros(num_problems, num_test_regimes)
    h_choice = zeros(Int, num_problems, num_test_regimes)
    nnets = Matrix(undef, num_problems, num_test_regimes)

    # Get model-based importance weights
    weights = Matrix(undef, num_train_regimes, num_test_regimes)
    for i = 1:num_train_regimes
        for j = 1:num_test_regimes
            weights[i, j] = report_density_ratio_ifm(x_train[i], sigma_train[i, :], sigma[j, :], theta, ifm)
        end
    end
    
    for j = 1:num_test_regimes
        if verbose
           println("[TEST REGIME ", j, "]")
           pb = ProgressBar(1:num_problems)
        else
           pb = 1:num_problems
        end
        for p = pb    
            h_choice[p, j], nnets[p, j] = supervised_train_covariateshift(x_train, y_trains[p], weights[:, j], num_hidden_space, 
                                                                          num_folds, rho_nn, max_iter_nn, false)
            y_hat[p, j] = mean(predict_multidata_nn(nnets[p, j], h_choice[p, j], [x_modelhat_montecarlo[j]])[1])
        end
    end

    return y_hat, nnets, h_choice, weights

end

####################################################
# BENCHMARK STEPS

""" `benchmark_generate_synthetic_groundtruth(
            x_data, sigma_data, sigma_all, sigma_scope, m, burn_in_ifm, num_problems,
            pa_x, num_hidden_dag_mean, num_hidden_dag_var, rho_dag, max_iter_dag,
            factors_x, num_levels_ifm_groundtruth, num_hidden_ifm_groundtruth, rho_ifm, max_iter_ifm,
            SAVE_PATH, GROUND_TRUTH_FILENAME, verbose = false)`

This function takes several real iid samples `x_data` of random vectors X, each sample `x_data[i]` coming from
regime `sigma_data[i, :]`. Each intervention variable assigment is given by `sigma_data[i, j]`, and takes
values in 0, 1, ..., `sigma_scope[j]`. Once we fit a model to this data, we generate `m` samples,
with a burn-in period of `burn_in_ifm` samples from the distribution of system X. Two models are fit to `x_data`,
`sigma_data`: a DAG with parent structure given by vector `pa_x` and factors which are conditionally Gaussian likelihoods
with mean given by a MLP with `num_hidden_dag_mean` hidden units and a variance given by a MLP with `num_hidden_dag_var` 
hidden units. Thi is fit to data using the Adam optimizer with parameter `rho_dag` and through `max_iter_dag` iterations;
the second model is an interventional factor model with factor structure given by vector `factors_x`, each corresponding
log-potential function parameterized by a MLP with `num_hidden_ifm` hidden units, fit by the Adam algorithm with
parameter `rho_ifm` by `max_iter_ifm` iterations. We then also generate `num_problems` output distributions of some
random variable Y given X, given by a Gaussian likelihood with mean given by a random linear function of X with a
tanh non-linearity. Finally, the ground truth parameters and Monte Carlo representations are saved in JDL2 file
with name `GROUND_TRUTH_FILENAME`, to be located in directory `SAVE_PATH`.
"""
function benchmark_generate_synthetic_groundtruth(
                x_data, sigma_data, sigma_all, sigma_scope, m, burn_in_ifm, num_problems,
                pa_x, num_hidden_dag_mean, num_hidden_dag_var, rho_dag, max_iter_dag,
                factors_x, num_levels_ifm_groundtruth, num_hidden_ifm_groundtruth, rho_ifm, max_iter_ifm,
                SAVE_PATH, GROUND_TRUTH_FILENAME, verbose = false)

    ###########################################################################################
    # Generate synthetic ground truth: Mediator layer
    #
    
    num_all_regimes = size(sigma_all, 1)

    # Create and fit a CausalDAGRegression model as synthetic ground truth to generate mediators
    groundtruth_dag = causal_dag_learning(x_data, sigma_data, pa_x, sigma_scope, num_hidden_dag_mean, num_hidden_dag_var, rho_dag, max_iter_dag, verbose)
    
    # Create and fit a (discretized) IFM model as synthetic ground truth to generate mediators
    groundtruth_ifm_theta, _, groundtruth_ifm, _, _, groundtruth_ifm_dx_support, _ = 
            learn_ifm(x_data, sigma_data, factors_x, sigma_scope, num_levels_ifm_groundtruth, num_hidden_ifm_groundtruth, rho_ifm, max_iter_ifm, verbose)

    # Now, sample large samples of mediator variables X from each of the 16 target regimes, so that we 
    # can later compute a Monte Carlo estimate of the ground truth expected values of outcome variable Y
    montecarlo_x_dag = sample_causal_dag(groundtruth_dag, sigma_all, m .* ones(Int, num_all_regimes), VERBOSE);
    montecarlo_x_ifm = Vector(undef, num_all_regimes)
    if VERBOSE
        println("Sampling from IFM ground truth...")
    end
    for i = 1:num_all_regimes
        if VERBOSE
            println("[REGIME ", i, "/", num_all_regimes, "]")
        end
        montecarlo_x_ifm[i] = sample_ifm_gibbs(m, burn_in_ifm, groundtruth_ifm, sigma_all[i, :], groundtruth_ifm_theta, groundtruth_ifm_dx_support, verbose)
    end

    ###########################################################################################
    # Generate synthetic ground truth: Outcome layer
    #
    
    y_signal_strenght = zeros(num_problems)
    lambda_true_dag = Vector(undef, num_problems)
    lambda_true_ifm = Vector(undef, num_problems)
    true_y_dag = Vector(undef, num_problems)
    true_var_y_dag = Vector(undef, num_problems)
    true_y_ifm = Vector(undef, num_problems)
    true_var_y_ifm = Vector(undef, num_problems)

    if VERBOSE
        println("Generating outcome problems...")
    end

    for p = 1:num_problems

        # Artificial outcome model ("Y" variable), controlled to have
        # empirical variance of 1 at the observational regime
        y_signal_strenght[p] = rand() * 0.2 + 0.6 # Signal to noise ratio, between 0.6 and 0.8
        y_var_error_true = 1. - y_signal_strenght[p]
        pre_lambda_true = rand(Normal(), num_x)
        pre_var_y = var(montecarlo_x_dag[1] * pre_lambda_true)
        lambda_true_dag[p] = pre_lambda_true ./ sqrt(pre_var_y) * sqrt(y_signal_strenght[p])
        pre_var_y = var(montecarlo_x_ifm[1] * pre_lambda_true)
        lambda_true_ifm[p] = pre_lambda_true ./ sqrt(pre_var_y) * sqrt(y_signal_strenght[p])

        # "True" expected values under all regimes of interest (in the three models), as estimated using Monte Carlo
        true_y_dag[p] = zeros(Float64, num_all_regimes, 1)
        true_var_y_dag[p] = zeros(Float64, num_all_regimes, 1)
        for i = 1:num_all_regimes
            dat = tanh.(montecarlo_x_dag[i] * lambda_true_dag[p]) + rand(Normal(), size(montecarlo_x_dag[i], 1)) .* sqrt(y_var_error_true)
            true_y_dag[p][i, 1], true_var_y_dag[p][i, 1] = mean(dat), var(dat)
        end
        true_y_ifm[p] = zeros(Float64, num_all_regimes, 1)
        true_var_y_ifm[p] = zeros(Float64, num_all_regimes, 1)
        for i = 1:num_all_regimes
            dat = tanh.(montecarlo_x_ifm[i] * lambda_true_ifm[p]) + rand(Normal(), size(montecarlo_x_ifm[i], 1)) .* sqrt(y_var_error_true)
            true_y_ifm[p][i, 1], true_var_y_ifm[p][i, 1] = mean(dat), var(dat)
        end

    end
    
    if verbose
        println("Saving ground truth...")
    end
    file = File(format"JLD2", SAVE_PATH * GROUND_TRUTH_FILENAME)
    save(file, "groundtruth_dag", groundtruth_dag,
                "groundtruth_ifm_theta", groundtruth_ifm_theta, "groundtruth_ifm", groundtruth_ifm, "groundtruth_ifm_dx_support", groundtruth_ifm_dx_support,
                "montecarlo_x_dag", montecarlo_x_dag, "montecarlo_x_ifm", montecarlo_x_ifm,
                "num_problems", num_problems, "y_signal_strenght", y_signal_strenght, 
                "lambda_true_dag", lambda_true_dag, "lambda_true_ifm", lambda_true_ifm,
                "true_y_dag", true_y_dag, "true_var_y_dag", true_var_y_dag,
                "true_y_ifm", true_y_ifm, "true_var_y_ifm", true_var_y_ifm)

    return groundtruth_dag, groundtruth_ifm_theta, groundtruth_ifm, groundtruth_ifm_dx_support, montecarlo_x_dag, montecarlo_x_ifm,
           y_signal_strenght, lambda_true_dag, lambda_true_ifm, true_y_dag, true_var_y_dag, true_y_ifm, true_var_y_ifm
    
end

""" `benchmark_generate_training_data(sigma_data, y_signal_strenght, lambda_true_dag, lambda_true_ifm,
                                      groundtruth_dag, groundtruth_ifm, groundtruth_ifm_theta, groundtruth_ifm_dx_support,
                                      sample_size, factor_baseline, SAVE_PATH, SYNTHETIC_TRAIN_FILENAME, verbose = false)`

This generates training data from two ground truth models parameterized by `y_signal_strenght`, `lambda_true_dag`,
`lambda_true_ifm`, `groundtruth_dag`, `groundtruth_ifm`, `groundtruth_ifm_theta`, `groundtruth_ifm_dx_support` as given
by the outcome of a call to `benchmark_generate_synthetic_groundtruth`. For each regime in `sigma_data` we generate
`sample_size` samples, with the first regime (as given by `sigma_data[1, :]`) gets a boost by a factor of `factor_baseline`,
if one wishes to interpret it as a "baseline" regime from which historical data is easily available. Results are saved
to a JDL2 file names `SYNTHETIC_TRAIN_FILENAME` in path `SAVE_PATH`.
"""
function benchmark_generate_training_data(sigma_data, y_signal_strenght, lambda_true_dag, lambda_true_ifm,
                                          groundtruth_dag, groundtruth_ifm, groundtruth_ifm_theta, groundtruth_ifm_dx_support,
                                          sample_size, factor_baseline, SAVE_PATH, SYNTHETIC_TRAIN_FILENAME, verbose = false)

    num_problems = length(y_signal_strenght)
    all_y_train_dag = Vector(undef, num_problems)
    all_y_train_ifm = Vector(undef, num_problems)

    if verbose
        println("Generating mediator training data...")
    end
    n_data = sample_size .* ones(Int, num_regimes); n_data[1] *= factor_baseline # Training sample size
    burn_in_ifm = 100
    train_dat_dag = sample_causal_dag(groundtruth_dag, sigma_data, n_data, VERBOSE)
    train_dat_ifm = Vector(undef, num_regimes)
    if verbose
        println("Sampling IFM training data...")
    end
    for i = 1:num_regimes
        if verbose
            println("Regime ", i)
        end
        train_dat_ifm[i] = sample_ifm_gibbs(n_data[i], burn_in_ifm, groundtruth_ifm, sigma_data[i, :], 
                                            groundtruth_ifm_theta, groundtruth_ifm_dx_support, verbose)
    end

    if verbose
        println("Generating output training data...")
        pb = ProgressBar(1:num_problems)
    else
        pb = 1:num_problems
    end

    for p = pb

        y_var_error_true = 1. - y_signal_strenght[p]

        # Generate synthetic Y data
        y_train_dag = Vector{Matrix{Float64}}(undef, num_regimes)
        for i = 1:num_regimes
            y_train_dag[i] = Matrix{Float64}(undef, n_data[i], 1)
            y_train_dag[i][:, 1] = tanh.(train_dat_dag[i] * lambda_true_dag[p]) + rand(Normal(), n_data[i]) .* sqrt(y_var_error_true)
        end

        y_train_ifm = Vector{Matrix{Float64}}(undef, num_regimes)
        for i = 1:num_regimes
            y_train_ifm[i] = Matrix{Float64}(undef, n_data[i], 1)
            y_train_ifm[i][:, 1] = tanh.(train_dat_ifm[i] * lambda_true_ifm[p]) + rand(Normal(), n_data[i]) .* sqrt(y_var_error_true)
        end

        # Collect

        all_y_train_dag[p], all_y_train_ifm[p] = y_train_dag, y_train_ifm

    end

    file = File(format"JLD2", SAVE_PATH * SYNTHETIC_TRAIN_FILENAME)
    save(file, "train_dat_dag", train_dat_dag, "train_dat_ifm", train_dat_ifm,
               "all_y_train_dag", all_y_train_dag, "all_y_train_ifm", all_y_train_ifm)

    return train_dat_dag, train_dat_ifm, all_y_train_dag, all_y_train_ifm

end

""" `benchmark_save_results(SAVE_PATH, LEARNED_Y_MODEL_FILENAME, LEARNED_RESULT_FILENAME,
                            true_y, true_var_y, y_hat_dag, y_hat_ifm_1, y_hat_ifm_2, y_hat_ifm_3, 
                            nnets_ifm_pred, ipw_ifm_pred, h_choice_ifm_pred, y_hat_blackbox)`

Save fitted output models to a JDL2 file named `LEARNED_Y_MODEL_FILENAME`, and estimated means
to CSV files starting with the name `LEARNED_RESULT_FILENAME`. For examples of what the other
inputs are, consult example script `workflow_examples.jl`.
"""
function benchmark_save_results(SAVE_PATH, LEARNED_Y_MODEL_FILENAME, LEARNED_RESULT_FILENAME,
                                true_y, true_var_y, y_hat_dag, y_hat_ifm_1, y_hat_ifm_2, y_hat_ifm_3, 
                                nnets_ifm_pred, ipw_ifm_pred, h_choice_ifm_pred, y_hat_blackbox)

    num_problems = size(y_hat_blackbox, 1)

    file = File(format"JLD2", SAVE_PATH * LEARNED_Y_MODEL_FILENAME)
    save(file, "y_hat_dag", y_hat_dag, "y_hat_ifm_1", y_hat_ifm_1, "y_hat_ifm_2", y_hat_ifm_2, 
               "y_hat_ifm_3", y_hat_ifm_3, "nnets_ifm_pred", nnets_ifm_pred, "ipw_ifm_pred", ipw_ifm_pred, "h_choice_ifm_pred", h_choice_ifm_pred,
               "y_hat_blackbox", y_hat_blackbox)

    true_mat = zeros(num_problems, num_all_regimes)
    [true_mat[i, :] = true_y[i] for i = 1:num_problems]
    true_var_mat = zeros(num_problems, num_all_regimes)
    [true_var_mat[i, :] = true_var_y[i] for i = 1:num_problems]

    CSV.write(SAVE_PATH * LEARNED_RESULT_FILENAME * "true.csv", DataFrame(true_mat, :auto), header=false)
    CSV.write(SAVE_PATH * LEARNED_RESULT_FILENAME * "blackbox.csv", DataFrame(y_hat_blackbox, :auto), header=false)
    CSV.write(SAVE_PATH * LEARNED_RESULT_FILENAME * "y_hat_dag.csv", DataFrame(y_hat_dag, :auto), header=false)
    CSV.write(SAVE_PATH * LEARNED_RESULT_FILENAME * "y_hat_ifm_1.csv", DataFrame(y_hat_ifm_1, :auto), header=false)
    CSV.write(SAVE_PATH * LEARNED_RESULT_FILENAME * "y_hat_ifm_2.csv", DataFrame(y_hat_ifm_2, :auto), header=false)
    CSV.write(SAVE_PATH * LEARNED_RESULT_FILENAME * "y_hat_ifm_3.csv", DataFrame(y_hat_ifm_3, :auto), header=false)
    CSV.write(SAVE_PATH * LEARNED_RESULT_FILENAME * "true_var_y.csv", DataFrame(true_var_mat, :auto), header=false)

end

""" `benchmark_report_statistics(sigma_all, true_y, true_var_y,
                                 y_hat_dag, y_hat_ifm_1, y_hat_ifm_2, y_hat_ifm_3, y_hat_blackbox,
                                 include_sigma, sigma_choice; y_hat_ifm = nothing, problem_example = 0)`

Report summary statistics on how well each model is doing compared to ground truth `true_y` in a selection
`include_sigma` of regimes from `sigma_all`. Ground truth `true_y` contains multiple examples. To print
a table of summary statistics for a particular example, indicate that with argument `problem_example`.
See script `workflow_examples.jl` for an example of usage.
"""
function benchmark_report_statistics(sigma_all, true_y, true_var_y,
                                     y_hat_dag, y_hat_ifm_1, y_hat_ifm_2, y_hat_ifm_3, y_hat_blackbox,
                                     include_sigma, sigma_choice; y_hat_ifm = nothing, problem_example = 0)

    num_problems = length(true_y)

    if isnothing(y_hat_ifm)
        y_hat_ifm = y_hat_ifm_1
    end

    rmse_dag, spearman_dag, prmse_dag = zeros(num_problems), zeros(num_problems), zeros(num_problems)
    rmse_ifm, spearman_ifm, prmse_ifm = zeros(num_problems), zeros(num_problems), zeros(num_problems)
    rmse_ifm1, spearman_ifm1, prmse_ifm1 = zeros(num_problems), zeros(num_problems), zeros(num_problems)
    rmse_ifm2, spearman_ifm2, prmse_ifm2 = zeros(num_problems), zeros(num_problems), zeros(num_problems)
    rmse_ifm3, spearman_ifm3, prmse_ifm3 = zeros(num_problems), zeros(num_problems), zeros(num_problems)
    rmse_blackbox, spearman_blackbox, prmse_blackbox = zeros(num_problems), zeros(num_problems), zeros(num_problems)

    for p = 1:num_problems

        rmse_dag[p] = sqrt(mean((true_y[p][include_sigma] - y_hat_dag[p, include_sigma]).^2))
        rmse_ifm[p] = sqrt(mean((true_y[p][include_sigma] - y_hat_ifm[p, include_sigma]).^2))
        rmse_ifm1[p] = sqrt(mean((true_y[p][include_sigma] - y_hat_ifm_1[p, include_sigma]).^2))
        rmse_ifm2[p] = sqrt(mean((true_y[p][include_sigma] - y_hat_ifm_2[p, include_sigma]).^2))
        rmse_ifm3[p] = sqrt(mean((true_y[p][include_sigma] - y_hat_ifm_3[p, include_sigma]).^2))
        rmse_blackbox[p] = sqrt(mean((true_y[p][include_sigma] - y_hat_blackbox[p, include_sigma]).^2))

        spearman_dag[p] = corspearman(true_y[p][include_sigma], y_hat_dag[p, include_sigma])
        spearman_ifm[p] = corspearman(true_y[p][include_sigma], y_hat_ifm[p, include_sigma])
        spearman_ifm1[p] = corspearman(true_y[p][include_sigma], y_hat_ifm_1[p, include_sigma])
        spearman_ifm2[p] = corspearman(true_y[p][include_sigma], y_hat_ifm_2[p, include_sigma])
        spearman_ifm3[p] = corspearman(true_y[p][include_sigma], y_hat_ifm_3[p, include_sigma])
        spearman_blackbox[p] = corspearman(true_y[p][include_sigma], y_hat_blackbox[p, include_sigma])

        prmse_dag[p] = mean((true_y[p][include_sigma] - y_hat_dag[p, include_sigma]).^2 ./ true_var_y[p][include_sigma])
        prmse_ifm[p] = mean((true_y[p][include_sigma] - y_hat_ifm[p, include_sigma]).^2 ./ true_var_y[p][include_sigma])
        prmse_ifm1[p] = mean((true_y[p][include_sigma] - y_hat_ifm_1[p, include_sigma]).^2 ./ true_var_y[p][include_sigma])
        prmse_ifm2[p] = mean((true_y[p][include_sigma] - y_hat_ifm_2[p, include_sigma]).^2 ./ true_var_y[p][include_sigma])
        prmse_ifm3[p] = mean((true_y[p][include_sigma] - y_hat_ifm_3[p, include_sigma]).^2 ./ true_var_y[p][include_sigma])
        prmse_blackbox[p] = mean((true_y[p][include_sigma] - y_hat_blackbox[p, include_sigma]).^2 ./ true_var_y[p][include_sigma])

    end

    if problem_example > 0
        result_frame = DataFrame([sigma_all[:, sigma_choice] true_y[problem_example] y_hat_blackbox[problem_example, :] y_hat_dag[problem_example, :] y_hat_ifm[problem_example, :]], :auto)
        result_frame[!, 1:length(sigma_choice)] = convert.(Int, result_frame[:, 1:length(sigma_choice)])
        rename!(result_frame, [:s2, :s4, :s7, :s9, :true_y, :blackbox, :dag, :ifm])
        println(result_frame)
        println()
        println("RMSE (blackbox)  : ", rmse_blackbox[problem_example])
        println("RMSE (DAG)       : ", rmse_dag[problem_example])
        println("RMSE (IFM)       : ", rmse_ifm[problem_example])
        println()
        println("RANKCORR (blackbox)  : ", spearman_blackbox[problem_example])
        println("RANKCORR (DAG)       : ", spearman_dag[problem_example])
        println("RANKCORR (IFM)       : ", spearman_ifm[problem_example])
        println()
        println("pRMSE (blackbox)  : ", prmse_blackbox[problem_example])
        println("pRMSE (DAG)       : ", prmse_dag[problem_example])
        println("pRMSE (IFM)       : ", prmse_ifm[problem_example])
        println()
    end
    
    println() 
    println("ALL RESULTS:")
    println("------------")
    println("RMSE (blackbox)   : ", mean(rmse_blackbox))
    println("RMSE (DAG)        : ", mean(rmse_dag))
    println("RMSE (IFM1)       : ", mean(rmse_ifm1))
    println("RMSE (IFM2)       : ", mean(rmse_ifm2))
    println("RMSE (IFM3)       : ", mean(rmse_ifm3))
    println()
    println("RANKCORR (blackbox)   : ", mean(spearman_blackbox))
    println("RANKCORR (DAG)        : ", mean(spearman_dag))
    println("RANKCORR (IFM1)       : ", mean(spearman_ifm1))
    println("RANKCORR (IFM2)       : ", mean(spearman_ifm2))
    println("RANKCORR (IFM3)       : ", mean(spearman_ifm3))
    println()
    println("pRMSE (blackbox)  : ", mean(prmse_blackbox))
    println("pRMSE (DAG)       : ", mean(prmse_dag))
    println("pRMSE (IFM1)      : ", mean(prmse_ifm1))
    println("pRMSE (IFM2)      : ", mean(prmse_ifm2))
    println("pRMSE (IFM3)      : ", mean(prmse_ifm3))
    println()
    
end

