#########################################################################
# Interventional Factor Model learning (IFM)

using ForwardDiff, StatsBase, Random
using Printf

include("data_structures.jl")

""" `create_ifm(factors::Vector{Vector{Int}}, sigma_scope::Vector{Int}, num_hidden::Int)`

This creates a IFM based on the information `factors` about which variables go in which factors,
the number of categories `sigma_scope` of the corresponding intervention variables, and the number of
hidden units `num_hidden` used to define each factor. 

It also compiles redundant information, such as which factors each variable belongs to,
and where to find the parameters of a respective factor at a respective single vector
that stacks up all parameters of the model. 
"""
function create_ifm(factors_x::Vector{Vector{Int}}, sigma_scope::Vector{Int}, num_hidden::Int)

    # Compile where to find each variable x

    num_x = maximum(maximum.(factors_x))
    num_f = length(factors_x)
    x_factors = Vector{Vector{Int}}(undef, num_x)
    x_factors_pos = zeros(Int, num_f, num_x)
    for i = 1:num_x
        found = zeros(Int, num_f)
        for f = 1:num_f
            found[f] = Int(i in factors_x[f])
            if found[f] == 1
                x_factors_pos[f, i] = findfirst(factors_x[f] .== i)
            end
        end
        x_factors[i] = findall(found .== 1)
    end

    # Create an index of where to find each parameter set for factor x regime combinations

    pos_f = 1
    param_idx = Vector{Matrix{UnitRange{Int64}}}(undef, num_f)
    for f = 1:num_f
        param_idx[f] = Matrix{UnitRange{Int64}}(undef, sigma_scope[f] + 1, 3)
        # Number of weights and number of biases of layer 1, followed by number of weights of layer 2, 
        # for a total of 3 sizes
        layer_lengths = [(length(factors_x[f])) * num_hidden, num_hidden, num_hidden]
        for s = 0:sigma_scope[f]
            for (l_id, layer_length) in enumerate(layer_lengths)
                param_idx[f][s + 1, l_id] = pos_f:(pos_f + layer_length - 1)
                pos_f += layer_length
            end
        end
    end

    return Ifm(num_x, num_f, factors_x, x_factors, x_factors_pos, sigma_scope, num_hidden, param_idx)

end


########################################################## 
# MAIN METHOD
##
# This uses pseudo-likelihood, while discretizing the data.

""" `learn_ifm(x_train::Vector{Matrix{Float64}}, sigma_train::Matrix, 
               factors_x::Vector{Vector{Int}}, sigma_scope::Vector{Int}, 
               num_levels::Int, num_hidden::Int, rho::Float64, max_iter::Int=1000, verbose::Bool=false)`

Train an interventional factor model (IFM) on a collection of continuous datasets `x_train`, where dataset `x_train[i]`
is collected under regime `sigma_train[i, :]`. The IFM is composed of factors `factor_x`, where each factor
has one categorical intervention variable associated with, taking values in 0, 1, 2, ..., `sigma_scope[i]`.
The IFM is fitted by pseudo-likelihood by first discretizing the data into `num_levels` levels. Each factor is 
given by a MLP with `num_hidden` hidden units, and fit by the Adam algorithm with parameter `rho` via `max_iter`
iterations.
"""
function learn_ifm(x_train::Vector{Matrix{Float64}}, sigma_train::Matrix, 
                   factors_x::Vector{Vector{Int}}, sigma_scope::Vector{Int}, 
                   num_levels::Int, num_hidden::Int, rho::Float64, max_iter::Int=1000, verbose::Bool=false)

    ifm = create_ifm(factors_x, sigma_scope, num_hidden)
    num_regimes = size(sigma_train, 1)
    
    dx_train, dx_train_code, dx_support, dx_buckets = pooled_uniform_discretize(x_train, num_levels)

    # Initialize theta
    theta_size = ifm.param_idx[end][end, end][end]
    theta = rand(Normal(), theta_size) / sqrt(theta_size)
    m_theta, v_theta = zeros(theta_size), zeros(theta_size)

    # Subsample sizes
    n_sample = Vector{Int}(undef, num_regimes)
    n_total = Vector{Int}(undef, num_regimes)
    sel_rows_all = Vector(undef, num_regimes)
    for i = eachindex(x_train)
        n_total[i] = size(x_train[i], 1)
        n_sample[i] = Int(round(n_total[i] / 100))
        sel_rows_all[i] = 1:n_total[i]
    end

    # Optimize
    fun = zeros(max_iter)
    verbose ? pb = ProgressBar(1:max_iter) : pb = 1:max_iter
    theta_trail = Vector(undef, max_iter)
    for iter in pb
        theta_trail[iter] = copy(theta)
        sel_rows = subsample(n_total, n_sample)
        #fun[iter], gr = ifm_pseudollik_discrete_autodiff(theta, dx_train, dx_train_code, dx_support, sigma_train, sel_rows, ifm)
        fun[iter], gr = ifm_pseudollik_discrete_manual(theta, dx_train, dx_train_code, dx_support, sigma_train, sel_rows, ifm)
        theta, m_theta, v_theta = adam(theta, gr, iter, m_theta, v_theta, rho)
        if verbose
            set_description(pb, string(@sprintf("LLIK: %.2f", fun[iter])))
        end
    end

    theta = ifm_pseudollik_discrete_pick_theta(theta_trail, dx_train, dx_train_code, dx_support, sigma_train, sel_rows_all, ifm, verbose)

    # Return
    return theta, fun, ifm, dx_train, dx_train_code, dx_support, dx_buckets

end

function ifm_pseudollik_discrete(theta, dx_train, dx_train_code, dx_support, sigma_train, sel_rows, ifm)

    score = 0.

    for i = eachindex(dx_train)

        x, x_code = dx_train[i][sel_rows[i], :], dx_train_code[i][sel_rows[i], :]
        sigma = sigma_train[i, :]
        n_total, n = size(dx_train[i], 1), length(sel_rows[i])
        score_i = 0.

        for x_id = 1:ifm.num_x

            num_support = length(dx_support[x_id])
            energy = Vector{Vector}(undef, num_support)
            for v = 1:num_support
                energy[v] = zeros(n)
            end

            for f = ifm.x_factors[x_id]
                # Regime
                s = sigma[f] + 1
                # Weights and biases of layer 1a
                w_1 = reshape(theta[ifm.param_idx[f][s, 1]], (length(ifm.factors_x[f]), ifm.num_hidden))
                b_1 = theta[ifm.param_idx[f][s, 2]]
                # Weights of layer 2
                w_2 = theta[ifm.param_idx[f][s, 3]]
                # Energy function for each level
                x_copy = x[:, x_id]
                for v = 1:num_support
                    x[:, x_id] .= dx_support[x_id][v]
                    energy[v] += sigmoid.(x[:, ifm.factors_x[f]] * w_1 .+ b_1') * w_2
                end
                x[:, x_id] = x_copy
            end

            for k = 1:n
                z = 0.
                for v = 1:num_support
                    z += exp(energy[v][k])
                end
                score_i += energy[x_code[k, x_id]][k] - log(z)
            end

        end

        score += score_i / n * n_total 

    end 

    return score

end

function ifm_pseudollik_discrete_autodiff(theta, dx_train, dx_train_code, dx_support, sigma_train, sel_rows, ifm)
    fun = ifm_pseudollik_discrete(theta, dx_train, dx_train_code, dx_support, sigma_train, sel_rows, ifm)
    negloss_g = theta_0 -> ForwardDiff.gradient(theta -> ifm_pseudollik_discrete(theta, dx_train, dx_train_code, dx_support, sigma_train, sel_rows, ifm), theta_0);
    gr = negloss_g(theta)
    return fun, gr
end

function ifm_pseudollik_discrete_manual(theta, dx_train, dx_train_code, dx_support, sigma_train, sel_rows, ifm)

    score = 0.
    gr = zeros(length(theta))

    for i = eachindex(dx_train)

        x, x_code = dx_train[i][sel_rows[i], :], dx_train_code[i][sel_rows[i], :]
        sigma = sigma_train[i, :]
        n_total, n = size(dx_train[i], 1), length(sel_rows[i])
        score_i, gr_i = 0., zeros(length(theta))

        for x_id = 1:ifm.num_x

            num_support = length(dx_support[x_id])
            energy = zeros(n, num_support)
            sig_linear = Vector{Matrix{Float64}}(undef, ifm.num_f)

            # Update energy function for each level
            for f = ifm.x_factors[x_id]

                # Regime
                s = sigma[f] + 1

                # Weights and biases of layers 1 and 2 
                w_1 = reshape(theta[ifm.param_idx[f][s, 1]], (length(ifm.factors_x[f]), ifm.num_hidden))
                b_1 = theta[ifm.param_idx[f][s, 2]]
                w_2 = theta[ifm.param_idx[f][s, 3]]
                
                # Energy function and other sufficient statistics
                px = ifm.x_factors_pos[f, x_id]
                x_copy = x[:, x_id]
                x[:, x_id] .= 0
                sig_linear[f] = x[:, ifm.factors_x[f]] * w_1 .+ b_1'
                for v = 1:num_support
                    energy[:, v] += sigmoid.(sig_linear[f] .+ dx_support[x_id][v] * w_1[px, :]') * w_2
                end
                x[:, x_id] = x_copy

            end

            # Normalizing constant
            exp_energy = exp.(energy)
            z = sum(exp_energy, dims = 2)
            for row = 1:n
                score_i += energy[row, x_code[row, x_id]] - log(z[row])
            end

            # Gradient update
            for f = ifm.x_factors[x_id]

                s = sigma[f] + 1
                w_1 = reshape(theta[ifm.param_idx[f][s, 1]], (length(ifm.factors_x[f]), ifm.num_hidden))
                w_2 = theta[ifm.param_idx[f][s, 3]]
                px = ifm.x_factors_pos[f, x_id]

                # Numerator
                sig_f = sigmoid.(sig_linear[f] + x[:, x_id] * w_1[px, :]')
                dsig_ft = sig_f .* (1 .- sig_f) .* w_2'
                xh_pos = 0
                for h = 1:ifm.num_hidden
                    for x_col = ifm.factors_x[f]
                        xh_pos += 1
                        gr_i[ifm.param_idx[f][s, 1][xh_pos]] += sum(dsig_ft[:, h] .* x[:, x_col])
                    end
                end
                gr_i[ifm.param_idx[f][s, 2]] += sum(dsig_ft, dims = 1)'
                gr_i[ifm.param_idx[f][s, 3]] += sum(sig_f, dims = 1)'

                # Denominator
                for v = 1:num_support
                    sig_f = sigmoid.(sig_linear[f] .+ dx_support[x_id][v] .* w_1[px, :]')
                    dsig_ft = sig_f .* (1 .- sig_f) .* w_2'
                    x_copy = x[:, x_id]
                    x[:, x_id] .= dx_support[x_id][v]
                    xh_pos = 0
                    for h = 1:ifm.num_hidden
                        for x_col = ifm.factors_x[f]
                            xh_pos += 1
                            gr_i[ifm.param_idx[f][s, 1][xh_pos]] -= sum((exp_energy[:, v] .* dsig_ft[:, h] .* x[:, x_col]) ./ z)
                        end
                    end
                    x[:, x_id] = x_copy
                    gr_i[ifm.param_idx[f][s, 2]] -= sum(exp_energy[:, v] .* dsig_ft ./ z, dims = 1)'
                    gr_i[ifm.param_idx[f][s, 3]] -= sum(exp_energy[:, v] .* sig_f ./ z, dims = 1)'    
                end

            end

        end

        score += score_i / n * n_total 
        gr += gr_i / n * n_total

    end 

    return score, gr

end

function ifm_condprob_discrete(theta, x_id, x, dx_support, sigma, ifm)

    num_support = length(dx_support[x_id])
    probs = zeros(num_support)
    x_copy = x[x_id]

    for f = ifm.x_factors[x_id]
        # Regime
        s = sigma[f] + 1
        # Weights and biases of layer 1
        w_1 = reshape(theta[ifm.param_idx[f][s, 1]], (length(ifm.factors_x[f]), ifm.num_hidden))
        b_1 = theta[ifm.param_idx[f][s, 2]]
        # Weights of layer 2
        w_2 = theta[ifm.param_idx[f][s, 3]]
        # Energy function for each level
        for v = 1:num_support
            x[x_id] = dx_support[x_id][v]
            probs[v] += sum(sigmoid.(w_1' * x[ifm.factors_x[f]] + b_1) .* w_2)
        end
    end
    x[x_id] = x_copy

    probs = exp.(probs .- maximum(probs))
    probs /= sum(probs)

    return probs

end

function ifm_pseudollik_discrete_pick_theta(theta_trail, dx_train, dx_train_code, dx_support, sigma_train, sel_rows_all, ifm, verbose = false)
    if verbose
        println("   ...revisiting trail (full model)...")
    end
    best_llik = -Inf
    best_theta = []
    trail_size = length(theta_trail)
    if trail_size > 50
        backlog = 49
    else
        backlog = 1
    end
    verbose ? pb = ProgressBar((trail_size - backlog):trail_size) : pb = (trail_size - backlog):trail_size
    for i = pb
        llik = ifm_pseudollik_discrete(theta_trail[i], dx_train, dx_train_code, dx_support, sigma_train, sel_rows_all, ifm)
        if llik > best_llik
            best_llik = llik
            best_theta = theta_trail[i]
        end
        if verbose
            set_description(pb, string(@sprintf("BEST LLIK: %.2f", best_llik)))
        end
    end
    return best_theta
end

""" `sample_ifm_gibbs(m::Int, burn_in::Int, ifm::Ifm, sigma::Vector{Int}, 
                      theta::Vector{Float64}, dx_support::Vector{Vector{Float64}}, 
                      verbose::Bool = false)`

Sample `m` samples with a period of `burn_in` burn-in iterations from an interventional factor model
`ifm` with parameters `theta` and discretization scheme `dx_support`. There will be a different sample
for each regime `sigma[i, :]`.
"""
function sample_ifm_gibbs(m::Int, burn_in::Int, ifm::Ifm, sigma::Vector{Int}, 
                         theta::Vector{Float64}, dx_support::Vector{Vector{Float64}}, 
                         verbose::Bool = false)
    
    x = zeros(m + burn_in, ifm.num_x)
    for x_id = 1:ifm.num_x
        x[1, x_id] = dx_support[x_id][sample(1:length(dx_support[x_id]))]
    end
    max_iter = m + burn_in

    verbose ? pb = ProgressBar(2:max_iter) : pb = 2:max_iter
    for iter in pb
        x[iter, :] = x[iter - 1, :]
        for x_id = 1:ifm.num_x
            probs = ifm_condprob_discrete(theta, x_id, x[iter, :], dx_support, sigma, ifm)
            x[iter, x_id] = wsample(dx_support[x_id], probs)
        end
        if verbose
            iter <= burn_in ? set_description(pb, "burn in") : set_description(pb, "collecting")
        end
    end
 
    return x[(burn_in + 1):end, :]

end
