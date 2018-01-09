module MultipleScatteringLearnMoments

using MultipleScattering
using LaTeXStrings
using JLD
using StatsBase
using GLMNet
using Plots 

# The types exported
export ML_model, OptionsML, FeatureMap

# exporting global variables
export fmmoments, fmmean, fmofm_full

# The functions exported
export setup_ml, fit_L2, fit_L1, test_linear, get_crossvalid_parameters, predict_linear 
export learning_saturation_curve_grid_volfrac_rad_crossvalidation, cross_valid_inner, load_moments

type ML_model
    fiterror::Float64   
    parameters::Array{Float64}
    yscale::Float64
    ysmean::Float64
    ys::Array{Float64}
    Xmean::Array{Float64}
    Xstd::Array{Float64}
    name::AbstractString
    nfeatures::Int64
    numlabels::Int64
    X::Array{Float64}
    kernelname::AbstractString    
    kernelparam::Float64    
    kernelapply::Function
end

type OptionsML
    aux::Array{Int64} # Auxiliary array, Used for selecting the moments in moments_feature_map.jl
    pcts::Array{Float64}  # The percentages used in the quantiles regression
    plotfigures::Bool
end

#The apply function needs to follow a very particular pattern. See stack_rows_td_responses for an example
type FeatureMap
    apply::Function
    get_numfeatures::Function
    name::AbstractString
end

#calculates and save the best lambdas for a given feature map using cross validation over the training set
#include("cross_valid_lambdas.jl")
include("cross_valid_inner.jl")
# Receives a training data as an array of tds and returns a linear model
include("setup_ml.jl")
#include("train_linear.jl")
include("fit_L2.jl")
include("fit_L1.jl")
# Receives a response and a linear model (defined by ML_model) and returns a predicted label
include("predict_linear.jl")
# Receives a test response data set and a linear model (defined by ML_model) and returns the average test error
include("test_linear.jl")
# Plots a heatmap of the testover over each labelled pair.
include("testerror_heatmap.jl")
# The feature map that only uses the mean response curve
include("mean_feature_map.jl")
# The feature map that uses centered moments of response simulation
include("moments_feature_map.jl");
# The feature map that uses all moments or a given selection of centered moments of response simulation
include("mofm_full_feature_map.jl")
# Definitions and functions for defining different Kernenls
include("kernels.jl");
# Inclues the data transformation functions, such as centering and scaling
include("data_transformations.jl");
# Misc
include("get_crossvalid_parameters.jl");
# Calculate the curve of the R^2 error on the test data as we progressively use more training data
# in old folder: include("frequency_saturation.jl") 
# in old folder: include("learning_saturation_curve_grid_volfrac_rad.jl")
include("learning_saturation_curve_grid_volfrac_rad_crossvalidation.jl")
# in old folder:include("feature_selection_L1.jl")
include("load_moments.jl")

end # module
