##########################################################################################
# other_learning.jl
# Generic auxiliary learning algorithms.
#

include("data_structures.jl")

using ForwardDiff
using XGBoost
using StatsBase, Distributions
using Printf

##########################################################################################
# GENERIC OPTIMIZATION AND GENERALLY USEFUL FUNCTIONS
#

""" `adam(w, g, t, m, v, rho)`

Perform one step of ADAM optimization with respect to parameter
`w`, with gradient `g`, at step count `t`, with statistics
`m` and `v`, and learning rate `rho`.
"""
function adam(w::Vector{Float64}, g::Vector{Float64}, t::Int,
              m::Vector{Float64}, v::Vector{Float64}, rho::Float64)
    beta_1 = 0.9
    beta_2 = 0.999
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * g.^2
    m_hat = m / (1 - beta_1^t)
    v_hat = v / (1 - beta_2^t)
    w += rho * m_hat ./ (sqrt.(v_hat) .+ 1.e-8)
    return w, m, v
end

""" `relu`
    
Rectified linear activation function.
"""
@inline relu(x) = (x < 0 ? 0 : x)

""" `sofplus`
    
Softplus activation function.
"""
@inline softplus(x) = log(1. + exp(x))

""" `sigmoid`
    
Sigmoid activation function.
"""
@inline sigmoid(x) = (1 / (1 + exp(-x)))

##########################################################################################
# GENERIC SUPERVISED LEARNING
#
# A wrapper for supervised learning that includes cross-validation of a mapping from `x`
# to `y` using `k` fold cross-validation.
#

function supervised_train(x, y, k = 10, verbose = false)

    n = length(y)
    shuffle_rows = randperm(n)
    folds = [collect(i:k:n) for i in 1:k]

    min_child_weight_range, max_depth_range = 1:5, 2:10
    num_h = length(min_child_weight_range) * length(max_depth_range)
    h, pos_h = Matrix{Int}(undef, num_h, 2), 1
    for c in min_child_weight_range
        for d in max_depth_range
            h[pos_h, :] = [c, d]
            pos_h += 1
        end
    end

    best_loss = Inf
    best_h = [0, 0]
    verbose ? pb = ProgressBar(1:num_h) : pb = 1:num_h
    for iter = pb
        for (iter, fold) = enumerate(folds)
            val_idx = shuffle_rows[fold]
            train_idx = setdiff(1:n, val_idx)
            x_train, x_val = x[train_idx, :], x[val_idx, :]
            y_train, y_val = y[train_idx], y[val_idx]
            model = xgboost((x_train, y_train), min_child_weight = h[iter, 1], max_depth = h[iter, 2], watchlist = (;))
            y_val_hat = XGBoost.predict(model, x_val)
            val_error = mean((y_val - y_val_hat).^2)
            if val_error < best_loss
                best_h = h[iter, :]
                best_loss = val_error
            end
            if verbose
                set_description(pb, string(@sprintf("Best h: %d, %d", best_h[1], best_h[2])))
            end    
        end
    end

    model = xgboost((x, y), min_child_weight = best_h[1], max_depth = best_h[2], watchlist = (;))
    err_var = var(y - XGBoost.predict(model, x))
    return model, err_var

end

function supervised_train(x, y, k, num_hidden_mean, num_hidden_var, rho, max_iter, verbose = false)

    n = length(y)
    shuffle_rows = randperm(n)
    folds = [collect(i:k:n) for i in 1:k]

    lambda_range = [0.01, 0.1, 1.]
    num_lambda = length(lambda_range)

    best_loss = Inf
    best_lambda = 0.
    for iter = 1:num_lambda
        if verbose
            println("[[Attempting λ = ", lambda_range[iter], "]]")
        end
        val_loss = 0.
        for (iter, fold) = enumerate(folds)
            val_idx = shuffle_rows[fold]
            train_idx = setdiff(1:n, val_idx)
            x_train, x_val = x[train_idx, :], x[val_idx, :]
            y_train, y_val = y[train_idx], y[val_idx]
            theta, _ = learn_neural_hregression(x_train, y_train, num_hidden_mean, num_hidden_var, lambda_range[iter], rho, max_iter, verbose)
            y_val_mean, y_val_var = predict_neural_hregression(theta, x_val, num_hidden_mean, num_hidden_var)
            val_loss += -sum(-0.5 .* log.(y_val_var) - 0.5 .* (y_val.- y_val_mean).^2 ./ y_val_var)
        end
        if val_loss < best_loss
            best_lambda = lambda_range[iter]
            best_loss = val_loss
            if verbose
                println("(IMPROVEMENT)")
            end
        end
    end

    if verbose
        println("[[CONCLUDING]]")
    end
    theta, _ = learn_neural_hregression(x, y, num_hidden_mean, num_hidden_var, best_lambda, rho, max_iter, verbose)
    return [num_hidden_mean, num_hidden_var, theta]

end

function supervised_prediction(x, model)
    return XGBoost.predict(model, x)
end

function supervised_prediction(x, num_hidden::Int)

    p = size(x, 2)

    w_pos, w_size = 0, x * num_hidden
    w_1 = resize(theta[(w_pos + 1):(w_pos + w_size)], p, num_hidden)
    w_pos += w_size
    w_size += p
    b_1 = theta[(w_pos + 1):(w_pos + w_size)]
    hidden_layer = tanh.(x * w_1 .+ b_1')

    w_pos += w_size
    w_size = p + 1
    w_2 = theta[(w_pos + 1):(w_pos + w_size)]
    b_2 = theta[w_pos + w_size + 1]
    y_mean = hidden_layer * w_2 .+ b_2

    return y_mean

end

########################################################
# METHODS FOR NEURAL HETEROSKEDASTIC REGRESSION
#

""" `neural_hregression_llik(theta, x, y, num_hidden, lambda)`

Gaussian log-likelihood for the regression of `y` on `x` with a multilayer perceptron
with `num_hidden_mean` hidden units for the mean and `num_hidden_var` hidden units for
the variance of `y` conditioned on `x`.
"""
function neural_hregression_llik(theta, x, y, num_hidden_mean, num_hidden_var, lambda)
    y_mean, y_var = predict_neural_hregression(theta, x, num_hidden_mean, num_hidden_var)
    return sum(-0.5 .* log.(y_var) - 0.5 .* (y - y_mean).^2 ./ y_var) + lambda * mean(theta.^2)
end

function predict_neural_hregression(theta, x, num_hidden_mean, num_hidden_var)

    n, p = size(x)
    w_pos = 0

    # Mean

    w_size = p * num_hidden_mean
    w_1 = reshape(theta[(w_pos + 1):(w_pos + w_size)], p, num_hidden_mean)
    w_pos += w_size
    w_size = num_hidden_mean
    b_1 = theta[(w_pos + 1):(w_pos + w_size)]
    hidden_layer = tanh.(x * w_1 .+ b_1')

    w_pos += w_size
    w_size = num_hidden_mean
    w_2 = theta[(w_pos + 1):(w_pos + w_size)]
    b_2 = theta[w_pos + w_size + 1]
    y_mean = hidden_layer * w_2 .+ b_2
    w_pos += w_size + 1

    # Variance

    w_size = p * num_hidden_var
    w_1 = reshape(theta[(w_pos + 1):(w_pos + w_size)], p, num_hidden_var)
    w_pos += w_size
    w_size = num_hidden_var
    b_1 = theta[(w_pos + 1):(w_pos + w_size)]
    hidden_layer = tanh.(x * w_1 .+ b_1')

    w_pos += w_size
    w_size = num_hidden_var
    w_2 = theta[(w_pos + 1):(w_pos + w_size)]
    b_2 = theta[w_pos + w_size + 1]
    y_var_log = hidden_layer * w_2 .+ b_2
    y_var = exp.(y_var_log)
    
    # (Penalized) log-likelihood

    return y_mean, y_var

end

function neural_hregression_llik_autodiff(theta, x, y, num_hidden_mean, num_hidden_var, lambda)
    fun = neural_hregression_llik(theta, x, y, num_hidden_mean, num_hidden_var, lambda)
    llik_g = theta_0 -> ForwardDiff.gradient(theta -> neural_hregression_llik(theta, x, y, num_hidden_mean, num_hidden_var, lambda), theta_0);
    gr = llik_g(theta)
    return fun, gr
end

function supervised_sample_hregression(model, x)
    n = size(x, 1)
    num_hidden_mean, num_hidden_var, theta = model
    y_mean, y_var = predict_neural_hregression(theta, x, num_hidden_mean, num_hidden_var)
    return y_mean + rand(Normal(), n) .* sqrt.(y_var)
end

function learn_neural_hregression(x, y, num_hidden_mean, num_hidden_var, lambda, rho, max_iter, verbose=false)

    # Subsample sizes
    n_total, p = size(x)
    n_sample = Int(round(length(y) / 100))

    # Optimize
    fun = zeros(max_iter)
    theta_size = (p + 1) * num_hidden_mean + (num_hidden_mean + 1) +
                 (p + 1) * num_hidden_var + (num_hidden_var + 1) 
    theta = rand(Normal(), theta_size) ./ sqrt(length(theta_size))
    m_theta, v_theta = zeros(theta_size), zeros(theta_size)

    verbose ? pb = ProgressBar(1:max_iter) : pb = 1:max_iter
    for iter in pb
        sel_rows = sample(1:n_total, n_sample, replace=false)
        fun[iter], gr = neural_hregression_llik_autodiff(theta, x, y, num_hidden_mean, num_hidden_var, lambda)
        theta, m_theta, v_theta = adam(theta, gr, iter, m_theta, v_theta, rho)
        if verbose
            set_description(pb, string(@sprintf("LLIK: %.2f", fun[iter])))
        end
    end

    # Return
    return theta, fun

end

##########################################################################################
# GENERIC NEURAL NETWORKS WITH MULTIPLE DATASETS AND WEIGHTED SAMPLES
#

function learn_multidata_nn(input_train, output_train, row_weights_train, num_hidden, rho_nn, max_iter_nn, verbose=false)

    num_regimes = length(input_train)
    num_input = size(input_train[1], 2)

    theta_size = (num_input + 1) * num_hidden + (num_hidden + 1)
    nn_theta = rand(theta_size) ./ sqrt(theta_size)
    m_theta, v_theta = zeros(theta_size), zeros(theta_size)
    fun = zeros(max_iter_nn)

    verbose ? pb = ProgressBar(1:max_iter_nn) : pb = 1:max_iter_nn
    for iter in pb
        #fun[iter], gr = negloss_multidata_nn_autodiff(nn_theta, num_hidden, input_train, output_train, row_weights_train)
        fun[iter], gr = negloss_multidata_nn_manual(nn_theta, num_hidden, input_train, output_train, row_weights_train)
        nn_theta, m_theta, v_theta = adam(nn_theta, gr, iter, m_theta, v_theta, rho_nn)
        if verbose
            set_description(pb, string(@sprintf("MSE: %.2f", -fun[iter])))
        end
    end

    return nn_theta, fun

end

function negloss_multidata_nn(nn_theta, num_hidden, input_train, output_train, row_weights_train)

    num_input = size(input_train[1], 2)
    negloss = 0.
    num_regimes = length(input_train)

    w_1 = reshape(nn_theta[1:(num_input * num_hidden)], num_input, num_hidden)
    param_pos = num_input * num_hidden
    b_1 = nn_theta[(param_pos + 1):(param_pos + num_hidden)]
    param_pos += num_hidden
    w_2 = nn_theta[(param_pos + 1):(param_pos + num_hidden)]
    b_2 = nn_theta[param_pos + num_hidden + 1]

    for i = 1:num_regimes
        y_hat = sigmoid.(input_train[i] * w_1 .+ b_1') * w_2 .+ b_2
        negloss -= sum((output_train[i] - y_hat).^2 .* row_weights_train[i])
    end

    return negloss

end

function negloss_multidata_nn_autodiff(nn_theta, num_hidden, input_train, output_train, row_weights_train)

    fun = negloss_multidata_nn(nn_theta, num_hidden, input_train, output_train, row_weights_train)
    negloss_g = theta_0 -> ForwardDiff.gradient(nn_theta -> negloss_multidata_nn(nn_theta, num_hidden, input_train, output_train, row_weights_train), theta_0);
    gr = negloss_g(nn_theta)
    return fun, gr

end

function negloss_multidata_nn_manual(nn_theta, num_hidden, input_train, output_train, row_weights_train)

    num_input = size(input_train[1], 2)
    theta_size = length(nn_theta)
    negloss, grad = 0., zeros(theta_size)
    num_regimes = length(input_train)

    w_1 = reshape(nn_theta[1:(num_input * num_hidden)], num_input, num_hidden)
    param_pos = num_input * num_hidden
    b_1 = nn_theta[(param_pos + 1):(param_pos + num_hidden)]
    param_pos += num_hidden
    w_2 = nn_theta[(param_pos + 1):(param_pos + num_hidden)]
    b_2 = nn_theta[param_pos + num_hidden + 1]

    for i = 1:num_regimes
        hidden = sigmoid.(input_train[i] * w_1 .+ b_1')
        y_hat = hidden * w_2 .+ b_2
        signal = 2 .* (output_train[i] - y_hat) .* row_weights_train[i]
        hidden_grad = hidden .* (1 .- hidden)
        param_pos = 0
        for k = 1:num_hidden
            for j = 1:num_input        
                param_pos += 1
                grad[param_pos] += sum(signal .* input_train[i][:, j] .* hidden_grad[:, k]) * w_2[k]
            end
        end
        grad[(num_input * num_hidden + 1):(num_input * num_hidden + num_hidden)] += sum(signal .* hidden_grad, dims=1)' .* w_2
        param_pos += num_hidden
        grad[(param_pos + 1):(param_pos + num_hidden)] += sum(signal .* hidden, dims=1)'
        grad[end] += sum(signal)
        negloss -= sum((output_train[i] - y_hat).^2 .* row_weights_train[i])
    end

    return negloss, grad

end

function predict_multidata_nn(nn_theta, num_hidden, input_test)

    num_input = size(input_test[1], 2)
    negloss = 0.
    num_regimes = length(input_test)
    y_hat = Vector(undef, num_regimes)

    w_1 = reshape(nn_theta[1:(num_input * num_hidden)], num_input, num_hidden)
    param_pos = num_input * num_hidden
    b_1 = nn_theta[(param_pos + 1):(param_pos + num_hidden)]
    param_pos += num_hidden
    w_2 = nn_theta[(param_pos + 1):(param_pos + num_hidden)]
    b_2 = nn_theta[param_pos + num_hidden + 1]

    for i = 1:num_regimes
        y_hat[i] = sigmoid.(input_test[i] * w_1 .+ b_1') * w_2 .+ b_2
    end

    return y_hat

end

##########################################################################################
# CAUSAL DAG LEARNING
#
# These are specialised methods for DAG learning with interventions. Conditional 
# distributions are given by conditional Gaussian likelihoods with a MLP mean and a
# MLP variance function.

""" `causal_dag_learning(x_train::Vector{Matrix{Float64}}, sigma_train::Matrix{Int}, 
                         pa_x::Vector{Vector{Int}}, sigma_scope::Vector{Int},
                         num_hidden_dag_mean::Int, num_hidden_dag_var::Int,
                        rho::Float64, max_iter::Int, verbose::Bool = false)`

This estimates a causal DAG of soft categorical interventions. A one-to-one
mapping between intervention variables and random variables is assumed, meaning
that each random variable `i` has some interventional variable `i` even if
it always fixed to the baseline value.

- `dat_train`: different datasets under different regimes
- `sigma_train`: the matrix of intervention levels, where `sigma_train[i, :]` contains the intervention
  levels of regime `i` corresponding to `sigma_train[i]`
- `pa_x`: random parents of each random variable i.e. `pa_x[i]` are the random
  parents of random variable `i`
- `sigma_scope`: number of categories taken by each interventional variable. As in the
  `InterventionalFactorModel` representation, each `sigma_scope[i]` represent the number
   of categories of intervention variable `i` on top of the baseline value of 0.
   So, `sigma_scope[i] == 1` means that intervention variable `i` takes two values.
   If `sigma_scope[i] == 0`, here it will mean that no counterfactuals are defined for
   random variable `i`.

The model for each random variable given its parents is given by fitting a conditional Gaussian
likelihood with mean given by a MLP with `num_hidden_dag_mean` units and variance given by a 
MLP with `num_hidden_dag_var` units, one for which level of intervention parent.
"""
function causal_dag_learning(x_train::Vector{Matrix{Float64}}, sigma_train::Matrix{Int}, 
                             pa_x::Vector{Vector{Int}}, sigma_scope::Vector{Int},
                             num_hidden_dag_mean::Int, num_hidden_dag_var::Int,
                             rho::Float64, max_iter::Int, verbose::Bool = false)

    if verbose
        println("Fitting DAG model...")
    end
    num_x = length(pa_x)
    regressions = Vector(undef, num_x)
    var_error = Vector{Vector{Float64}}(undef, num_x)

    for j = 1:num_x

        if verbose
            println("[DAG LEARNING VARIABLE ", j, "]")
        end
        regressions[j] = Vector(undef, sigma_scope[j] + 1)
        var_error[j] = Vector{Float64}(undef, sigma_scope[j] + 1)

        for f = 0:sigma_scope[j]

            # First, get all datasets compatible with the interventional
            # parent being set to `f`
            sel_dat_idx = findall(sigma_train[:, j] .== f)
            if length(sel_dat_idx) == 0
                continue
            end
            sel_dat = Matrix{Float64}(undef, 0, num_x)
            for i = eachindex(sel_dat_idx)
                sel_dat = vcat(sel_dat, x_train[sel_dat_idx[i]])
            end

            # Now, fit model with a multilayer perceptron with Gaussian likelihood and
            # heteroskedastic noise.
            if !isempty(pa_x[j])
                input = sel_dat[:, pa_x[j]]
                output = sel_dat[:, j]
                regressions[j][f + 1] = supervised_train(input, output, 2, num_hidden_dag_mean, num_hidden_dag_var, rho, max_iter, verbose)
            else
                var_error[j][f + 1] = var(sel_dat[:, j])
            end

        end

    end

    return CausalDAGRegression(num_x, sigma_scope, pa_x, regressions, var_error)

end

""" `sample_causal_dag(model::CausalDAGRegression, sigma::Matrix{Int}, n::Vector{Int}, verbose::Bool = false)`

Sample different datasets with from a `CausalDAGRegression` model following regimes `sigma` and
sample sizes `n`.
"""                
function sample_causal_dag(model::CausalDAGRegression, sigma::Matrix{Int}, n::Vector{Int}, verbose::Bool = false)

    if verbose
        println("Sampling from DAG model...")
    end
    num_regimes = length(n)
    x_train = Vector{Matrix{Float64}}(undef, num_regimes)
    sampler = Normal()
    causal_order = get_partial_order(model.pa_x)

    verbose ? pb = ProgressBar(1:num_regimes) : pb = 1:num_regimes
    for i = pb
        x_train[i] = Matrix{Float64}(undef, n[i], model.num_x)
        for j = causal_order   
            f = sigma[i, j] + 1
            if !isempty(model.pa_x[j])
                input = x_train[i][:, model.pa_x[j]]
                x_train[i][:, j] = supervised_sample_hregression(model.regressions[j][f], input)
            else
                x_train[i][:, j] = sqrt(model.error_var[j][f]) .* rand(sampler, n[i])
            end
        end
    end

    return x_train

end

##########################################################################################
# BLACKBOX LEARNING OF INTERVENTION OUTCOMES
#
# Learns a mapping from sigma to y.

""" `blackbox_learning(y_train::Vector{Matrix{Float64}}, sigma_train::Matrix{Int}, sigma_eval::Matrix{Int})`

Learn a blackbox mapping from interventions to output, return response function evaluated at `sigma_eval`
(the default being all possible combinations of interventions when they are categorical).
"""
function blackbox_learning(y_train::Vector{Matrix{Float64}}, sigma_train::Matrix{Int}, sigma_eval::Matrix{Int})
    
    s_dat_f, s_dat_y = build_supervised_data(y_train, sigma_train)
    blackbox_model, _ = supervised_train(s_dat_f, s_dat_y[:, 1], 10, false)
    e_y_hat = supervised_prediction(sigma_eval, blackbox_model)
    return e_y_hat, blackbox_model

end

########################################################
# METHODS FOR IPW AND COVARIATE SHIFT LEARNING

function supervised_train_covariateshift(inputs, outputs, row_weights, 
                                         num_hidden_space, num_folds, rho_nn, max_iter_nn, verbose = false)

    num_regimes = length(inputs)
    shuffle_rows, folds = Vector(undef, num_regimes), Vector(undef, num_regimes)
    n = zeros(Int, num_regimes)
    for i = 1:num_regimes
        n[i] = length(outputs[i])
        shuffle_rows[i] = randperm(n[i])
        folds[i] = [collect(j:num_folds:n[i]) for j in 1:num_folds]
    end

    num_hyper = length(num_hidden_space)
    best_loss = Inf
    best_h = 0.

    for h_iter = 1:num_hyper

        if verbose
            println("[[Attempting hidden units = ", num_hidden_space[h_iter], "]]")
        end

        val_loss = 0.

        for iter = 1:num_folds

            input_train, input_val = Vector(undef, num_regimes), Vector(undef, num_regimes)
            output_train, output_val = Vector(undef, num_regimes), Vector(undef, num_regimes)
            row_weights_train, row_weights_val = Vector(undef, num_regimes), Vector(undef, num_regimes)

            for i = 1:num_regimes
                fold = folds[i][iter]
                val_idx = shuffle_rows[i][fold]
                train_idx = setdiff(1:n[i], val_idx)
                input_train[i], input_val[i] = inputs[i][train_idx, :], inputs[i][val_idx, :]
                output_train[i], output_val[i] = outputs[i][train_idx], outputs[i][val_idx]
                row_weights_train[i] = row_weights[i][train_idx]
                row_weights_val[i] = row_weights[i][val_idx]
            end

            nn_theta, _ = learn_multidata_nn(input_train, output_train, row_weights_train, num_hidden_space[h_iter], rho_nn, max_iter_nn, verbose)
            output_val_mean = predict_multidata_nn(nn_theta, num_hidden_space[h_iter], input_val)
            for i = 1:num_regimes
                val_loss += sum((output_val[i] - output_val_mean[i]).^2 .* row_weights_val[i])
            end

        end

        if val_loss < best_loss
            best_h = num_hidden_space[h_iter]
            best_loss = val_loss
            if verbose
                println("(IMPROVEMENT)")
            end
        end

    end

    if verbose
        println("[[CONCLUDING]]")
    end
    nn_theta, _ = learn_multidata_nn(inputs, outputs, row_weights, best_h, rho_nn, max_iter_nn, verbose)
    return [best_h, nn_theta]

end

""" `report_density_ratio_ifm(dat::Matrix{Float64}, sigma_de::Vector{Int[]}, sigma_nu::Vector{Int}, 
                              theta::Vector{Float64}, ifm::Ifm)`

For a given dataset `dat` collected under regime `sigma_de` following an interventional factor model
`ifm` with parameter vector `theta`, return the density ratio of the density given by regime `sigma_nu`
divided by the density of regime `sigma_de`.
"""
function report_density_ratio_ifm(dat::Matrix{Float64}, sigma_de::Vector{Int}, sigma_nu::Vector{Int}, 
                                  theta::Vector{Float64}, ifm::Ifm)

    n = size(dat, 1)
    pre_ratio = zeros(n)

    for f = 1:ifm.num_f
        
        s_de, s_nu = sigma_de[f] + 1, sigma_nu[f] + 1
        if s_de == s_nu
            continue
        end

        w_1 = reshape(theta[ifm.param_idx[f][s_de, 1]], (length(ifm.factors_x[f]), ifm.num_hidden))
        b_1 = theta[ifm.param_idx[f][s_de, 2]]
        w_2 = theta[ifm.param_idx[f][s_de, 3]]
        energy_de = sigmoid.(dat[:, ifm.factors_x[f]] * w_1 .+ b_1') * w_2

        w_1 = reshape(theta[ifm.param_idx[f][s_nu, 1]], (length(ifm.factors_x[f]), ifm.num_hidden))
        b_1 = theta[ifm.param_idx[f][s_nu, 2]]
        w_2 = theta[ifm.param_idx[f][s_nu, 3]]
        energy_nu = sigmoid.(dat[:, ifm.factors_x[f]] * w_1 .+ b_1') * w_2

        pre_ratio += energy_nu - energy_de

    end

    ratio = exp.(pre_ratio .- maximum(pre_ratio))
    ratio ./= sum(ratio)

    return ratio

end

