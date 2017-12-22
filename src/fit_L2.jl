#function for fitting a linear regression. Takes as input
#  tds::Array{ResponseData}      Training data set
#  fm::FeatureMap                Feature map  as defined in ML_models.jl
#  options                       Contains the regularization parameters and other optional arguments
#  kernalname                    Optional kernal name. kernelname = "" means no kernel and kernelname = "gauss" uses a gaussian kernel.
function fit_L2(ml::ML_model, options, kernelname, kernelparam, lambda) #,
    # Getting number of variables
    if(kernelname!="")
        nvars = size(ml.X)[1]; #if a kernel is used, we work in the dual, where the number of variables is equal to the #data
    else
        nvars = size(ml.X)[2];
    end
    ## Solving linear least squares
    lambda_norm = lambda/ml.numlabels;
    # Apply Kernel Transform
    ml.kernelparam = kernelparam;
    # Getting Kernel functions
    kernelmatrixapply, kernelapply = get_kernel_function(kernelname, kernelparam); 
    ml.kernelparam  = kernelparam;   
    ml.kernelapply  = kernelapply;
    kX = kernelmatrixapply(ml.X);
    ## Solving min 1/2|| kX w_r - y_r||_2^2 + lambda_r/2 ||w_r||_2^2 + 1/2|| kX w_v - y_v||_2^2 + lambda_v/2 ||w_v||_2^2
    A = kX'*kX;
    # solve for (X^TX+lambda_r I)wrs = X^T yrs
    A[:] = A+lambda_norm*eye(nvars);
    b = kX'*ml.ys;
    # Solve positive definite linear least squares system for radiuses
    LAPACK.posv!('U',A,b );
    ml.parameters = b;
    ml.fiterror = norm(kX*ml.parameters-ml.ys)/norm(ml.ys);
    if(options.plotfigures)
        println("Fitting ", ml.name)
        println(floor(1000.0*ml.fiterror)/10.0, "% absolute train error")
    end
    return ml
end
