#function for fitting a linear regression. Takes as input
#  tds::Array{ResponseData}      Training data set
#  fm::FeatureMap                Feature map  as defined in ML_models.jl
#  options                       Contains the regularization parameters and other optional arguments
# NOTE: You can not use a kernal combined with an L1 regularizor since we can no longer use the "kernel trick". But we can still use  feature maps.

function fit_L1(ml::ML_model, options,  λs) 
    ## Setting up the identity kernel, just so the model is consistent with testing code
    kernelmatrixapply, kernelapply = get_kernel_function("", 0.0); 
    ml.kernelparam  = 0.0;   
    ml.kernelapply  = kernelapply;
 
    ## Solving min 1/2|| ml.X w - y||_2^2 + lambda ||w||_1 
    # Note that intercept =false because the intercept has already been modelled in X and y !
    if(λs == "GLMNet")
        cv= glmnetcv(ml.X,ml.ys, intercept =false);   # Let GLMNet choose the lambda grid        
    else
        if(λs == 0.0)
           λs =  [ 2.0^-16.0,  2.0^-8.0,  2.0^-4.0, 2.0^-2.0, 2.0^-1.0, 1.0,  2.0^1.0, 2.0^2.0, 2.0^4.0]./ml.numlabels;
        end
        cv= glmnetcv(ml.X,ml.ys; lambda =λs, intercept =false);        
    end
    indbest = indmin(cv.meanloss);
    ml.parameters =  cv.path.betas[:, indbest];
    ml.fiterror = norm(ml.X*ml.parameters-ml.ys)/norm(ml.ys);
    if(options.plotfigures)
        println("Fitting ", ml.name)
        println(floor(1000.0*ml.fiterror)/10.0, "% absolute train error")
    end
    return ml
end


    #Failed attempt with Lasso package. Produces only NaNs and throws no error :/
    #path = fit(LassoPath,ml.X,ml.ys) ;
    #ml.parameters = coef(path; select=:CV1se); # Gets a good coeficient?