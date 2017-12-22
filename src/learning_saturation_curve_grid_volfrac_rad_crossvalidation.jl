# Calculates and plots the learning/test curves as we increase the number of grids points in the volfrac X Radius space. 
function learning_saturation_curve_grid_volfrac_rad_crossvalidation(train::Array{StatisticalMoments{Float64}}, test::Array{StatisticalMoments{Float64}}, fm::FeatureMap, options, kernelname::AbstractString, output_type::AbstractString,kernelparams, crossnum,lambdas, numparts, dataname::AbstractString, datatestname::AbstractString)

  pairs = unique([ [td.label[2], td.label[1] ] for td in train]); # pairs of (radius, volume)
  lpr = length(pairs);
  batchsize = convert(Int64,floor(lpr/convert(Float64,numparts)));
  Rsqds = zeros(numparts);
  #Reshuffling the data
  s = sample(1:lpr, lpr,replace=false);
  pairs = pairs[s];   # Reshuffle the data
  # Get best cross valid lambdas and kernelparameter
  lambda, kernelparam = get_crossvalid_parameters(fm,kernelname,output_type, options, dataname);   
  for k =1:numparts #Adding on more pairs in each iteration
    trainselection = 1:batchsize*k;  println("Training with ",batchsize*k, " points" )
    if(k==numparts) trainselection = 1:lpr end # get all data on last batch
    trainpairs = pairs[trainselection];
    #Get the testpair coresponding to the crossnum th section
    trainbatch = filter(td -> ([td.label[2],td.label[1]] in trainpairs), train);
    #
    lambda, kernelparam=     cross_valid_inner(trainbatch, fm, options, kernelname, kernelparams,crossnum,lambdas, output_type, false, dataname)
    ml = setup_ml(trainbatch, fm,  kernelname,options, output_type);
    fit_L2(ml, options,  kernelname, kernelparam, lambda);
    Rsqd = test_linear(test,datatestname, fm,  ml,options,output_type);         
    Rsqds[k] =Rsqd;
  end
  ml = setup_ml(train, fm,  kernelname,options, output_type);
  savename ="$(ml.name)-learncurve-crossvalid"
  default_path = "../data/"
  println("saving the learning curves in $(default_path)$(savename).jld")
  save("$(default_path)$(savename).jld", "Rsqds",Rsqds)
  # Plotting learning saturation curve
  plot((1:1:numparts)./numparts ,Rsqds,line = :line, xlabel = "fraction of train data",
  ylabel = string(L"$R^2$ of the ", output_type),label = "", ylims = (-1,1), grid = false)
  #grid("off")
  savefig("../figures/$(savename).pdf");
end
