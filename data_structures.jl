#############################################################################################
# Primary data structures
#
# This encodes a library of useful data structures.

# This defines a causal DAG model over continuous variables where
# each variable is conditionally Gaussian with variance and mean indexed
# by its intervention parent. In this data structure, there is an
# one-to-one correspondence betweeen interventions and random variables,
# and intervention variables are always categorical.

struct CausalDAGRegression

    num_x::Int                          # Number of random variables
    sigma_scope::Vector{Int}            # Scope of each interventional variable
    pa_x::Vector{Vector{Int}}           # Random variable parents of each random variable vertex, if DAG
    regressions::Vector                 # Expected value of random variables given parents for each regime
    error_var::Vector{Vector{Float64}}  # Variance of random variables given parents for each regime

end

# This defines an interventional factor model where each factor is parameterize
# by a MLP with num_hidden hidden units. Each MLP has a single latent layer with a sigmoid
# activation function, with a linear activation function and no bias term for the output layer.

struct Ifm

    num_x::Int                                   # Number of random variables
    num_f::Int                                   # Number of factors
    factors_x::Vector{Vector{Int}}               # factors_x[f] lists the variable ids in factor f
    x_factors::Vector{Vector{Int}}               # x_factors[i] lists the factors where variable id i belongs to
    x_factors_pos::Matrix{Int}                   # x_factors_pos[f, i] indicates the position of variable in in factors_x[f]
    sigma_scope::Vector{Int}                     # sigma_scope[i] indicates that sigma_i takes values in 0, 1, ..., sigma_scope[i]
    num_hidden::Int                              # Number of hidden units in the MLP parameterization of each factor
    param_idx::Vector{Matrix{UnitRange{Int64}}}  # param_idx[f][s, l] indicates indices of the l-th layer of the regime s MLP for factor f
end


