function cross_valid_inner(train::Array{StatisticalMoments{Float64}}, fm::FeatureMap, options::OptionsML, kernelname::AbstractString, kernelparams::Array{Float64}, crossnum::Int64, lambdas::Array{Float64}, output_type::AbstractString, saveresults::Bool, dataname::AbstractString)
  # crossnum  the number of parts the training data is divided into to perform the cross validation
  # kernelparams is a grid of kernel parameters
  numlam = length(lambdas);
  if(kernelname =="")
    numkernelparams = 1;
  else
    numkernelparams = length(kernelparams);
  end
  lambdasScore = zeros(numlam,numkernelparams);
  pairs = unique([ [td.label[2], td.label[1] ] for td in train]); # pairs of (radius, volume)
  lpr = length(pairs);
  s = sample(1:lpr, lpr,replace=false); # Calculat a reshuffling of the the set 1, ..., lpr
  pairs = pairs[s];   # Reshuffle the data
    for k =1:crossnum # selecting the kth training and test sets.
#         println("testing ",k, "th cross validation")
        divisor  = convert(Int64,floor(lpr/crossnum));
        testselection = (1+divisor*(k-1)):divisor*k;
        trainpairs = pairs[testselection];
        #Get the testpair coresponding to the crossnum th section
        testtemp = filter(td -> ([td.label[2],td.label[1]] in trainpairs), train);
        #Use all remaining data as the training set
        traintemp= filter(td -> !([td.label[2],td.label[1]] in trainpairs), train);
        #Setting up ml model
        ml = setup_ml(traintemp, fm,  kernelname,options, output_type);
        for i =1:numlam
            #println("Testing lambdas = (",lambdas[i],",",lambdas[js],")");
            lambda = lambdas[i];
            for ks = 1: numkernelparams
              kernelparam =   kernelparams[ks];
              fit_L2(ml, options,  kernelname, kernelparam, lambda);
              Rsqd= test_linear(testtemp, dataname, fm,  ml,options,output_type);  
              lambdasScore[i,ks]=lambdasScore[i,ks]+Rsqd;
            end
         end
  end
  #get indexes of best pair of lambdas
  inds = findn(lambdasScore.==lambdasScore[indmax(lambdasScore)]);
  lambdabest  = lambdas[inds[1][1]];
  klparambest =   kernelparams[inds[2][1]];
#     println("inds[1] = $(inds[1][1])  and inds[2] =$(inds[2][1])")
    if(saveresults)
      savename = string(get_model_name(fm,kernelname,output_type,options),"-",dataname,"-crossvalid");
      default_path = "../data/"
      println("best lambda score = $(round(lambdasScore[inds[1][1],inds[2][1]]/(Float64(crossnum)),3)) was with lambda = $(lambdabest) and klparameter = $(klparambest)");
      println("saving the best lambda and kernalparameter in $(default_path)$(savename).jld")
      save("$(default_path)$(savename).jld", "lambdabest",lambdabest, "klparambest", klparambest,"lambdasScorebest",lambdasScore[inds[1][1],inds[2][1]]/(Float64(crossnum)));
    end
  return  lambdabest[1], klparambest[1]
end
