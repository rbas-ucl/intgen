###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# EXPERIMENTAL WORKFLOW: 
#
# In this script, we learn data, generate synthetic data with ground truth following two
# independence models (a DAG model or a factor graph model), then compare how different 
# methods behave with a variety of synthetic outcome models.
#
# Two data examples are provided, Sachs et al. and DREAM, as descripted in the companion
# manuscript.

###########################################################################################
# Setup

include("ifm.jl")
include("util.jl")
include("other_learning.jl")
include("benchmark_tools.jl")

VERBOSE = true                 # Print progress. Will cause issues in a Jupyter notebook, because of Progress bars.
SAVE_PATH = "experiments/"     # Where to save results, if applicable
LOAD_GROUND_TRUTH = true       # Don't regenerate ground truth, load from file
LOAD_LEARNED_MODELS = true     # Don't regenerate learned models, load from file
LOAD_SYNTHETIC_TRAIN = true    # Don't regenerate synthetic training data, load from file

USE_IFM_3 = false              # Turn this to true if IFM3 is to be used. It's very slow though.

CASE_STUDY_SACHS = 1           # Use Sachs et al.         
CASE_STUDY_DREAM = 2           # Use DREAM

USE_DAG_GROUNDTRUTH = 1        # Use a DAG model to create ground truth 
USE_IFM_GROUNDTRUTH = 2        # Use a (deep energy) IFM to create ground truth

case_study = CASE_STUDY_SACHS           # Choice of data example
mode_groundtruth = USE_IFM_GROUNDTRUTH  # Choice of ground truth model

###########################################################################################
# Load empirical data, including structures of interest
#

""" `load_workflow_sachs()`

Loads the Sachs et al. case study.
"""
function load_workflow_sachs()

    # File names of relevance

    GROUND_TRUTH_FILENAME = "sachs_groundtruth.jld2"    # File containing ground truth
    SYNTHETIC_TRAIN_FILENAME = "sachs_synthtrain.jld2"  # File containing synthetic training data

    # Raw mediator data ("X" measurements) on the 5 regimes selected in Sachs et al.
    my_path = "dat/"
    file_names = ["1. cd3cd28.csv", 
                  "3. cd3cd28+aktinhib.csv", 
                  "4. cd3cd28+g0076.csv",
                  "5. cd3cd28+psitect.csv",
                  "6. cd3cd28+u0126.csv"]
    x_data = Vector{Matrix{Float64}}(undef, length(file_names))
    for (i, file_name) in enumerate(file_names)
        # Log-scale makes things easier here
        x_data[i] = log.(readdlm(my_path * file_name, ',', Float64; header=true)[1])
    end
    group_normalize!(x_data) # The normalization is across all 5 regimes, so no data file
                             # will end up with zero mean/unit standard deviation

    # Interventional data, and combination of them by using a function from the included code
    sigma_data = zeros(Int, 5, 11)
    sigma_data[2, 7] = 1 # Akt-inhibitor set to 1, all the others at 0
    sigma_data[3, 9] = 1 # G0076 set to 1, all the others at 0
    sigma_data[4, 4] = 1 # Psitectorigenin set to 1, all the others at 0
    sigma_data[5, 2] = 1 # U0126 set to 1, all the others at 0
    sigma_train = sigma_data
    num_regimes = size(sigma_data)[1]

    # Causal structure given by Sachs et al.: DAG and factor graph model variations
    num_x = 11
    sigma_scope = zeros(Int, num_x)
    for i in (7, 9, 4, 2) sigma_scope[i] = 1; end # Each sigma is binary, maximum value in sigma_scope[i] is 1
    sigma_all = build_categorical_design_matrix(sigma_scope) # All 16 combinations of sigma variables
    num_all_regimes = size(sigma_all, 1)

    # DAG
    pa_x = Vector{Vector{Int}}(undef, num_x)
    pa_x[1]  = [8, 9]
    pa_x[2]  = [1, 8, 9]
    pa_x[3]  = [5]
    pa_x[4]  = [3, 5]
    pa_x[5]  = []
    pa_x[6]  = [2, 8]
    pa_x[7]  = [5, 6, 8]
    pa_x[8]  = [9]
    pa_x[9]  = [3, 4]
    pa_x[10] = [8, 9]
    pa_x[11] = [8, 9]

    # IFM
    factors_x = Vector{Vector{Int}}(undef, num_x)
    for i in 1:num_x
        factors_x[i] = [i; pa_x[i]]
    end

    return x_data, sigma_data, num_x, sigma_scope, sigma_all, sigma_train, num_regimes, num_all_regimes, pa_x, factors_x,
           GROUND_TRUTH_FILENAME, SYNTHETIC_TRAIN_FILENAME

end

""" `load_workflow_dream()`

Loads the DREAM case study.
"""
function load_workflow_dream()

    # File names of relevance
    GROUND_TRUTH_FILENAME = "dream_groundtruth.jld2"    # File containing ground truth
    SYNTHETIC_TRAIN_FILENAME = "dream_synthtrain.jld2"  # File containing synthetic training data
    
    # Raw mediator data ("X" measurements) on the 11 regimes generated from DREAM
    my_path = "dat/InSilicoSize10-Ecoli1/"
    file_names = ["observational.csv", 
                "knockdowns_0.csv", 
                "knockdowns_1.csv",
                "knockdowns_2.csv",
                "knockdowns_3.csv",
                "knockdowns_4.csv",
                "knockdowns_5.csv",
                "knockdowns_6.csv",
                "knockdowns_7.csv",
                "knockdowns_8.csv",
                "knockdowns_9.csv"]
    x_data = Vector{Matrix{Float64}}(undef, length(file_names))
    for (i, file_name) in enumerate(file_names)
        x_data[i] = readdlm(my_path * file_name, ',', Float64; header=true)[1]
        x_data[i] = x_data[i][:, 2:end]
    end
    group_normalize!(x_data) # The normalization is across all 5 regimes, so no data file
                             # will end up with zero mean/unit standard deviation

    # Interventional data, and combination of them by using a function from the included code
    sigma_data = zeros(Int, 11, 10)
    for i = 2:11 
        sigma_data[i, i - 1] = 1
    end
    sigma_train = sigma_data
    num_regimes = size(sigma_data, 1)

    # Causal structure given by DREAM: DAG and factor graph variations
    num_x = 10
    sigma_scope = ones(Int, num_x)
    sigma_all = build_categorical_design_matrix(sigma_scope, 2) # All 56 combinations of pairwise sigma variables
    num_all_regimes = size(sigma_all, 1)

    # DAG
    pa_x = Vector{Vector{Int}}(undef, num_x)
    pa_x[1]  = [2]
    pa_x[2]  = []
    pa_x[3]  = [2]
    pa_x[4]  = [3, 9]
    pa_x[5]  = [3, 8, 9]
    pa_x[6]  = [3]
    pa_x[7]  = [3, 8, 10]
    pa_x[8]  = []
    pa_x[9]  = []
    pa_x[10] = []

    # IFM
    factors_x = Vector{Vector{Int}}(undef, num_x)
    for i in 1:num_x
        factors_x[i] = [i; pa_x[i]]
    end

    return x_data, sigma_data, num_x, sigma_scope, sigma_all, sigma_train, num_regimes, num_all_regimes, pa_x, factors_x,
           GROUND_TRUTH_FILENAME, SYNTHETIC_TRAIN_FILENAME

end

if case_study == CASE_STUDY_SACHS
    x_data, sigma_data, num_x, sigma_scope, sigma_all, sigma_train, num_regimes, 
    num_all_regimes, pa_x, factors_x, GROUND_TRUTH_FILENAME, SYNTHETIC_TRAIN_FILENAME = 
        load_workflow_sachs()
elseif case_study == CASE_STUDY_DREAM
    x_data, sigma_data, num_x, sigma_scope, sigma_all, sigma_train, num_regimes, 
    num_all_regimes, pa_x, factors_x, GROUND_TRUTH_FILENAME, SYNTHETIC_TRAIN_FILENAME = 
        load_workflow_dream()
else
    println("Not a recognizable case study!")
    Base.exit(1)
end

###########################################################################################
# GENERATE SYNTHETIC GROUND TRUTH
#
# Two different ground truth simulators for the mediators are generated. One which is
# based on a DAG, the other on a (discretized) IFM. 
#
# Output models are common, and consist of a tanh transformation of a linear function of 
# mediators, with additive noise.
#

if !LOAD_GROUND_TRUTH

    num_hidden_dag_mean, num_hidden_dag_var = 10, 10 # Number of hidden units for mean and variance of each factor in DAG modl
    rho_dag, max_iter_dag = 0.1, 250                 # Adam optimization parameters for DAG model learning
    
    num_levels_ifm_groundtruth = 20                  # Number of discretization levels for IFM
    num_hidden_ifm_groundtruth = 15                  # Number of hidden units for each factor in the IFM
    rho_ifm, max_iter_ifm = 0.1, 200                 # Adam optimization parameters for IFM learning

    num_problems = 100                               # Number of output problems to be created
    m = 25000                                        # Number of samples for Monte Carlo approximation of ground truth functionals
    burn_in_ifm = 100                                # Burn in for IFM MCMC sampling

    groundtruth_dag, groundtruth_ifm_theta, groundtruth_ifm, groundtruth_ifm_dx_support, montecarlo_x_dag, montecarlo_x_ifm,
           y_signal_strenght, lambda_true_dag, lambda_true_ifm, true_y_dag, true_var_y_dag, true_y_ifm, true_var_y_ifm = 
                benchmark_generate_synthetic_groundtruth(x_data, sigma_data, sigma_all, sigma_scope, m, burn_in_ifm, num_problems,
                    pa_x, num_hidden_dag_mean, num_hidden_dag_var, rho_dag, max_iter_dag,
                    factors_x, num_levels_ifm_groundtruth, num_hidden_ifm_groundtruth, rho_ifm, max_iter_ifm,
                    SAVE_PATH, GROUND_TRUTH_FILENAME, VERBOSE)

else

    if VERBOSE
        println("Loading ground truth...")
    end
    file = File(format"JLD2", SAVE_PATH * GROUND_TRUTH_FILENAME)
    extract(load(file))

end

###########################################################################################
# Generate training data
#
# We generate a variety of problems based on the same idea of a tahn + linear MLP synthetic
# model for the outcome variable.

if !LOAD_SYNTHETIC_TRAIN

    sample_size, factor_baseline = 500, 10
    train_dat_dag, train_dat_ifm, all_y_train_dag, all_y_train_ifm =
        benchmark_generate_training_data(sigma_data, y_signal_strenght, lambda_true_dag, lambda_true_ifm,
                                         groundtruth_dag, groundtruth_ifm, groundtruth_ifm_theta, groundtruth_ifm_dx_support,
                                         sample_size, factor_baseline, SAVE_PATH, SYNTHETIC_TRAIN_FILENAME, VERBOSE)

else

    file = File(format"JLD2", SAVE_PATH * SYNTHETIC_TRAIN_FILENAME)
    extract(load(file))

end

###########################################################################################
# Run comparisons
#
# Now, run experiments comparing IFM approaches to a DAG model and a blackbox.

# Select ground truth model structure

if mode_groundtruth == USE_DAG_GROUNDTRUTH
    x_train = train_dat_dag
    y_trains = all_y_train_dag
    true_y = true_y_dag
    true_var_y = true_var_y_dag
    LEARNED_X_MODEL_FILENAME = "sachs_synthlearned_dag.jld2"
    LEARNED_Y_MODEL_FILENAME = "sachs_synthlearned_dag_y.jld2"
    LEARNED_RESULT_FILENAME = "sachs_dagtruth_results_"
elseif mode_groundtruth == USE_IFM_GROUNDTRUTH
    x_train = Vector{Matrix{Float64}}(undef, num_regimes)
    [x_train[i] = train_dat_ifm[i] for i = 1: num_regimes]
    y_trains = all_y_train_ifm
    true_y = true_y_ifm
    true_var_y = true_var_y_ifm
    LEARNED_X_MODEL_FILENAME = "sachs_synthlearned_ifm.jld2"
    LEARNED_Y_MODEL_FILENAME = "sachs_synthlearned_ifm_y.jld2"
    LEARNED_RESULT_FILENAME = "sachs_ifmtruth_results_"
else
    println("Not a recognizable ground truth selection!")
    Base.exit(1)
end

# Optimization and sampling options

rho = 0.1                    # Adam IFM optimizer hyperparameter, the higher the "jumpier"
max_iter = 150               # Number of optimization iterations for IFM
num_hidden_ifm = 25          # Number of hidden units for IFM
num_levels = 20              # Number of discretization levels for IFM

num_hidden_dag_mean, num_hidden_dag_var = 10, 10 # Number of hidden units for DAG (non-linear Gaussian heteroscedastic) models
rho_dag, max_iter_dag = 0.1, 250                 # Adam optimization parameters for DAG

burn_in, m = 1000, 5000      # MCMC burn_in and number of iterations for sampler

# Learning mediator models

if !LOAD_LEARNED_MODELS

    # Learn

    theta_ifm_hat, fun_ifm_hat, ifm_hat, dx_train, dx_train_code, dx_support, dx_buckets  = 
            learn_ifm(x_train, sigma_train, factors_x, sigma_scope, num_levels, num_hidden_ifm, rho, max_iter, VERBOSE)

    learned_dag_model = causal_dag_learning(x_train, sigma_train, pa_x, sigma_scope, num_hidden_dag_mean, num_hidden_dag_var, rho_dag, max_iter_dag, VERBOSE)

    # Sample

    x_dag_hat_montecarlo = sample_causal_dag(learned_dag_model, sigma_all, m .* ones(Int, num_all_regimes), VERBOSE)

    x_ifm_hat_montecarlo = Vector{Matrix{Float64}}(undef, num_all_regimes)
    if VERBOSE
        println("Sampling predictive IFM samples...")
    end
    for i = 1:num_all_regimes
        if VERBOSE
            println("[REGIME ", i, "]")
        end
        x_ifm_hat_montecarlo[i] = sample_ifm_gibbs(m, burn_in, ifm_hat, sigma_all[i, :], theta_ifm_hat, dx_support, VERBOSE);
    end

    file = File(format"JLD2", SAVE_PATH * LEARNED_X_MODEL_FILENAME)
    save(file, "learned_dag_model", learned_dag_model, "theta_ifm_hat", theta_ifm_hat, "fun_ifm_hat", fun_ifm_hat, "ifm_hat", ifm_hat, 
            "num_hidden_ifm", num_hidden_ifm, "x_dag_hat_montecarlo", x_dag_hat_montecarlo, "x_ifm_hat_montecarlo", x_ifm_hat_montecarlo, 
            "dx_train", dx_train, "dx_train_code", dx_train_code, "dx_support", dx_support, "dx_buckets", dx_buckets)

else

    file = File(format"JLD2", SAVE_PATH * LEARNED_X_MODEL_FILENAME)
    extract(load(file))

end

########################  Go through all prediction problems now - first, arrange data

x_trains = Vector(undef, num_problems)
dx_trains = Vector(undef, num_problems)

for p = 1:num_problems
    x_trains[p] = x_train   # Same data for the mediators at each problem
    dx_trains[p] = dx_train # Discretized version of mediators for IFM approaches
end

######################## Solve DAG

if VERBOSE
    println("Finalizing DAG inference!")
end
y_hat_dag = batch_mediator_predict(x_dag_hat_montecarlo, sigma_all, x_trains, y_trains, VERBOSE)

######################## Solve blackbox

if VERBOSE
    println("Finalizing blackbox inference!")
    pb = ProgressBar(1:num_problems)
else
    pb = 1:num_problems
end
y_hat_blackbox = zeros(num_problems, num_all_regimes)
for p = pb
    y_hat_blackbox[p, :], _ = blackbox_learning(y_trains[p], sigma_train, sigma_all)
end

######################## Solve IFM

if VERBOSE
    println("Finalizing IFM inference!")
end
y_hat_ifm_1 = batch_mediator_predict(x_ifm_hat_montecarlo, sigma_all, dx_trains, y_trains, VERBOSE)
y_hat_ifm_2, _ = batch_mediator_predict(sigma_all, x_train, y_trains, sigma_train, ifm_hat, theta_ifm_hat, VERBOSE)
num_hidden_space_ifm, num_folds_ifm, rho_nn, max_iter_nn = [5, 10], 2, 0.1, 100
if USE_IFM_3
    # IFM 3 is very slow as implemented. If you really want to use it, set USE_IFM_3 to true
    y_hat_ifm_3, nnets_ifm_pred, h_choice_ifm_pred, ipw_ifm_pred = batch_mediator_predict(x_ifm_hat_montecarlo, sigma_all,
                                                                        dx_train, y_trains, sigma_train, ifm_hat,
                                                                        theta_ifm_hat, num_hidden_space_ifm, num_folds_ifm, rho_nn, max_iter_nn, VERBOSE)
else
    # Dummy values to keep the code simpler
    y_hat_ifm_3 = y_hat_ifm_1
    nnets_ifm_pred = h_choice_ifm_pred = ipw_ifm_pred = []
end

######################## Save results

if VERBOSE
    println("Finalizing IFM inference!")
end
benchmark_save_results(SAVE_PATH, LEARNED_Y_MODEL_FILENAME, LEARNED_RESULT_FILENAME,
                       true_y, true_var_y, y_hat_dag, y_hat_ifm_1, y_hat_ifm_2, y_hat_ifm_3, 
                       nnets_ifm_pred, ipw_ifm_pred, h_choice_ifm_pred, y_hat_blackbox)

######################## Report summary statistics

include_sigma = findall(sum(sigma_all, dims = 2) .> 1)
sigma_choice = [2, 4, 7, 9]
benchmark_report_statistics(sigma_all, true_y, true_var_y, y_hat_dag, y_hat_ifm_1, y_hat_ifm_2, y_hat_ifm_3, y_hat_blackbox,
                            include_sigma, sigma_choice, y_hat_ifm = y_hat_ifm_1, problem_example = 1)

if VERBOSE
    println("DONE")
end
