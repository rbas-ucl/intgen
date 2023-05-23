##########################################################################################
# util.jl
# Generic auxiliary functions
#

include("data_structures.jl")

using StatsBase
using Distributions
using ProgressBars

""" `extract(d)`

Put elements of a dictionary in the global scope.
From https://stackoverflow.com/questions/40833527/export-keys-and-values-of-dictionary-as-variable-name-and-value
"""
function extract(d)
    expr = quote end
    for (k, v) in d
        push!(expr.args, :($(Symbol(k)) = $v))
    end
    eval(expr)
    return
end

""" `log_sum_exp`

Given a vector `m` as input, it does `log(sum(exp.(m)))` in a way that is more
numerically stable.
"""
function log_sum_exp(m)
    max_m = maximum(m)
    return log(sum(exp.(m .- max_m))) + max_m 
end

""" `group_normalize(dats::Vector{Matrix{Float64}})`

A quick and dirty way of scaling and centering the columns of a set of
datasets. This is done by taking the averages of the averages and the averages
of the standard deviations for each column, across all datasets, and applying the
same shift/rescaling based on those two (unweighted) aggregations. 
"""
function group_normalize!(dats::Vector{Matrix{Float64}})
    p = size(dats[1], 2)
    num_dats = length(dats)
    m, s = zeros(p), zeros(p)
    for i in 1:p
        for j = 1:num_dats
            m[i] += mean(dats[j][:, i])
            s[i] += sqrt(var(dats[j][:, i]))
        end
        m[i] /= num_dats
        s[i] /= num_dats
        for j = 1:num_dats
            dats[j][:, i] .-= m[i]
            dats[j][:, i] ./= s[i]
        end
    end
    return m, s
end

function group_normalize!(dats::Vector{Matrix{Float64}}, m::Vector{Float64}, s::Vector{Float64})
    p = size(dats[1], 2)
    num_dats = length(dats)
    for i in 1:p
        for j = 1:num_dats
            dats[j][:, i] .-= m[i]
            dats[j][:, i] ./= s[i]
        end
    end
end

""" `get_partial_order(pa::Vector{Vector{Int}})`

Given a vector of vectors `pa`, where `pa[i]` is a vector with the index of
the parents of a vertex `i` in a DAG, this returns some ordering of
`1`, `2`, ..., `length(pa)` that is compatible with the DAG ordering.
"""
function get_partial_order(pa::Vector{Vector{Int}})

    p = length(pa)
    pa_m = zeros(Int, p, p)
    for i = 1:p
        pa_m[i, pa[i]] .= 1
    end
    order = Vector{Int}(undef, p)
    used_up = zeros(Int, p)

    for i = 1:p
        num_p = sum(pa_m, dims=2)
        j = findfirst((num_p .== 0) .* (used_up .== 0))[1]
        order[i] = j
        pa_m[:, j] .= 0
        used_up[j] = 1
    end

    return order

end

""" `build_supervised_data(y_train::Vector{Matrix{Float64}}, sigma_train::Matrix{Float64})`

From outcome variables at different regimes (`y_train[i]` for regime `i`), 
build a matrix of inputs and outputs to be used by standard supervised learning methods. In
particular, the inputs will be the regime vectors `sigma_train`.
"""
function build_supervised_data(y_train::Vector{Matrix{Float64}}, sigma_train::Matrix)

    num_regimes, num_sigma = size(sigma_train)
    input = Matrix{Float64}(undef, 0, num_sigma)
    output = Matrix{Float64}(undef, 0, 1) 

    for i = 1:num_regimes
        n = size(y_train[i])[1]
        input = vcat(input, repeat(sigma_train[i, :]', n))
        output = vcat(output, y_train[i])
    end

    return input, output

end

""" `build_categorical_design_matrix(sigma_scope::Matrix{Int})`

Build table `sigma` containing all possible combinations of (categorical) intervention levels.
Recall that `sigma_scope[i]` encodes that intervention `i` takes values in 0, 1, 2, `sigma_scope[i]`.
A value of 0 in `sigma_scope[i]` just means variable `i` has no intervention defined on it. 
"""
function build_categorical_design_matrix(sigma_scope::Vector{Int})

    num_f = length(sigma_scope)
    @assert sigma_scope[1] >= 0
    f = reshape([0:sigma_scope[1];], sigma_scope[1] + 1, 1)

    for i = 2:num_f
        n = size(f)[1]
        f_i = Matrix{Int}(undef, 0, i) 
        for j = 0:sigma_scope[i]
            f_i = vcat(f_i, hcat(f, repeat([j], n)))
        end
        f = f_i
    end
    
    return f

end

""" `build_categorical_design_matrix(sigma_scope::Matrix{Int}, nonzeros::Int)`

Build table `sigma` containing all possible combinations of (categorical) intervention levels.
Recall that `sigma_scope[i]` encodes that intervention `i` takes values in 0, 1, 2, `sigma_scope[i]`.
A value of 0 in `sigma_scope[i]` just means variable `i` has no intervention defined on it. 

In this version, no more than `nonzeros` entries are allowed to be nonzeros. The implementation here,
however, is fairly naive! It just calls the original method and then throws away everything that doesn't
obey the constraint. Not scalable even if `nonzeros` is small.
"""
function build_categorical_design_matrix(sigma_scope::Vector{Int}, nonzeros::Int)

    f = build_categorical_design_matrix(sigma_scope)
    keep = findall(sum(f .> 0, dims = 2)[:, 1] .<= nonzeros)
    return f[keep, :]

end

""" `build_vector_data(x_dat::Vector{Matrix}, sigma_dat::Matrix)`

Build vector-of-vectors representation for datasets `x_dat` and `sigma_dat`.
"""
function build_vector_data(x_dat::Vector{Matrix{Float64}}, sigma_dat::Matrix)
    
    v_x = Vector{Vector{Float64}}(undef, 0)
    v_sigma = Vector{Vector}(undef, 0)
    for i = eachindex(x_dat)
        n = size(x_dat[i], 1)
        new_x = [x_dat[i][j, :] for j in 1:n]
        v_x = vcat(v_x, new_x)
        sigma_now = repeat(sigma_dat[i, :]', n)
        new_sigma = [sigma_now[j, :] for j in 1:n]
        v_sigma = vcat(v_sigma, new_sigma)
    end

    return v_x, v_sigma

end

function subsample(dat::Vector{Matrix{Float64}}, n_sample::Vector{Int})
    n_dat = length(dat)
    dat_out = Vector{Matrix{Float64}}(undef, n_dat)
    for i = eachindex(dat)
        sel_dat = sample(1:size(dat[i], 1), n_sample[i], replace=false)
        dat_out[i] = dat[i][sel_dat, :]
    end
    return dat_out
end

function subsample(n::Vector{Int}, n_sample::Vector{Int})
    n_dat = length(n)
    sel_rows = Vector{Vector{Int}}(undef, n_dat)
    for i = eachindex(n)
        sel_rows[i] = sort(sample(1:n[i], n_sample[i], replace=false))
    end
    return sel_rows
end

""" `pooled_uniform_discretize(dat::Vector{Matrix{Float64}})`

This function goes through every variable in the collection of datasets `dat` and discretize it by uniform binning
on the support of the union of datasets, using `num_levels` levels of discretization.
"""
function pooled_uniform_discretize(dat::Vector{Matrix{Float64}}, num_levels::Int)

    num_dat = length(dat)
    p = size(dat[1], 2)
    d_dat = Vector{Matrix{Float64}}(undef, num_dat)
    d_dat_code = Vector{Matrix{Int}}(undef, num_dat)
    sup = Vector{Vector{Float64}}(undef, p)
    buckets = Vector{Vector{Float64}}(undef, p)

    for i = 1:num_dat
        n, p = size(dat[i])
        d_dat[i] = Matrix{Float64}(undef, n, p)
        d_dat_code[i] = Matrix{Int}(undef, n, p)
    end

    for i = 1:p

        l_x, u_x = minimum(dat[1][:, i]), maximum(dat[1][:, i])
        for j = 2:num_dat
            l_x, u_x = minimum([l_x; dat[j][:, i]]), maximum([u_x; dat[j][:, i]])
        end
        gap = (u_x - l_x) / (num_levels - 1)
        buckets[i] = Vector((l_x - gap / 2):gap:(u_x + gap / 2))
        if length(buckets[i]) == num_levels
            buckets[i] = [buckets[i]; u_x + gap / 2]
        end
        sup[i] = buckets[i][2:end] .- gap / 2

        for j = 1:num_dat
            n = size(dat[j], 1)
            x_j = dat[j][:, i]
            for k = 1:n
                d_dat_code[j][k, i] = findfirst(buckets[i] .> x_j[k]) - 1
                d_dat[j][k, i] = sup[i][d_dat_code[j][k, i]]
            end
        end

    end

    return d_dat, d_dat_code, sup, buckets

end

function apply_discretization(dat::Matrix{Float64}, sup::Vector{Vector{Float64}}, buckets::Vector{Vector{Float64}})

    n, p = size(dat)
    d_dat = Matrix{Float64}(undef, n, p)
    d_dat_code = Matrix{Int}(undef, n, p)

    for i = 1:p
        x_i = dat[:, i]
        for k = 1:n
            if buckets[i][end] < x_i[k]
                d_dat_code[k, i] = length(sup[i])
            elseif buckets[i][1] > x_i[k]
                d_dat_code[k, i] = 1
            else    
                d_dat_code[k, i] = findfirst(buckets[i] .> x_i[k]) - 1
            end
            d_dat[k, i] = sup[i][d_dat_code[k, i]]
        end
    end

    return d_dat, d_dat_code

end

function empirical_discrete_frequencies(d_dat, sup)
    num_support = length(sup)
    counts = zeros(num_support)
    for i = 1:num_support
        counts[i] = sum(d_dat .== sup[i])
    end
    counts ./= sum(counts)
end

""" `space_combo(n, k)`

Generate `(k + 1)^n` arrays of length `n`, which contains all combinations
of strings with characters going from `0` to `k`.
"""
function space_combo(n, k)

    if n == 1 
        result = zeros(Int, k, 1)
        result[:, 1] = 1:k
        return result
    end

    previous = space_combo(n - 1, k)
    n_previous = size(previous, 1)
    result = zeros(Int, k^n, n)

    pos = 0
    for i = 1:n_previous
        for j = 1:k
            pos += 1
            result[pos, 1:end-1] = previous[i, :]
            result[pos, end] = j
        end    
    end

    return result

end
