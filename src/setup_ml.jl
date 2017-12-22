#function for setting up the linear machine model. Takes training
#  train::Array{StatisticalMoments{Float64}}         Training data set of moments, se StatisticalMoments.jl for definition of moments
#  fm::FeatureMap                Feature map  as defined in ML_models.jl
#  options                       Contains the regularization parameters and other optional arguments
#  kernalname                    Optional kernal name. kernelname = "" means no kernel and kernelname = "gauss" uses a gaussian kernel.
function setup_ml(train::Array{StatisticalMoments{Float64}}, fm::FeatureMap,kernelname::AbstractString,  options, output_type::AbstractString) #,
    # Get feature selection if there is one
    if (isempty(options.aux))
        options.aux = get_feature_L1(fm,output_type);
    end

    #Get all volfractions and particle radiuses
    volfracs= union([td.label[1] for td in train]);
    radiuses = union([td.label[2] for td in train]);
    numlabels = length(volfracs)*length(radiuses);
    #Initialize data matrix.  Each row is a row concatentation of all simulations of a single labels pair.
    featuresize = fm.get_numfeatures(train[1],options);
    X = zeros(numlabels,featuresize);
    ys =  zeros(numlabels,1);
    rowcounts = 1;
    # Initialize Machine Learning model
    ml = ML_model( 0, [0], 0.0, 0.0,[0], [0], [0.0], fm.name,featuresize,numlabels,[0], kernelname,0.0,x->x);
    ml.name = get_model_name(fm,kernelname,output_type,options);
#     if(addftL1toname) #If features were selected by L1, adding f to name
#         ml.name = string( ml.name, "-f")
#     end
    #Build data matrix.  Each row is a row concatentation of all simulations of a single labels pair.
    for mom in train
        #filling in each row of X with a feature vector
        fm.apply(X, rowcounts, mom,options);
        vl = mom.label[1] ;
        rd = mom.label[2] ;
        if(output_type=="radius")
              ys[rowcounts] = rd;
        elseif(output_type=="volumefraction")
              ys[rowcounts] = vl;
        else
              ys[rowcounts] = vl/rd^2;
        end
        rowcounts = rowcounts+1;
    end
    # Data transformations: Centering, scaling, exponential on positive terms
    X = X_transform(X, ml); # println("yvs before undo_transform: ",yvs)
    ys =y_transform(ys , ml);#    println("yvs after undo_transform: ",yvs)
    # Crop data
    s = sample(1:rowcounts-1, rowcounts-1,replace=false);
    X = X[s,:];  # clipping off  the zero rows and re-shuffling
    ys = ys[s];   # clipping off  the zero rows and re-shuffling

    ml.X = X;   # reshuffle the data at this point to avoid data leakage bias?
    ml.ys =ys;
    if(options.plotfigures)
        println("(#data, #features): (",size(ml.X)[1], ", ", size(ml.X)[2],")")
    end
    return ml;
end
