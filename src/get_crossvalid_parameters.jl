function get_crossvalid_parameters(fm::FeatureMap,  kernelname::AbstractString,output_type::AbstractString,options, dataname::AbstractString)
  savename = string(get_model_name(fm,kernelname,output_type,options),"-",dataname,"-crossvalid")
  default_path = "../data/"
  lambdabest =1.0;
  klparambest =1.0;
  #repeat =1 means we should repeat all calculations even if there is a saved output already
  try
    lambdabest, klparambest = load("$(default_path)$(savename).jld","lambdabest", "klparambest");
    println("found ", "$(default_path)$(savename).jld with lambdabest ",lambdabest, " and kernelparam ", klparambest)
  catch loaderror
    println(loaderror)
    println("No best lambdabest for $(fm.name)$(kernelname)")
  end
  return lambdabest, klparambest
end

function get_feature_L1(fm::FeatureMap,output_type::AbstractString)
  savename = string(fm.name,"-",output_type ,"-featureL1");
  default_path = "../data/"
  #repeat =1 means we should repeat all calculations even if there is a saved output already
  features = [];
  try
    features = load("$(default_path)$(savename).jld","features");
    println("found  features for $(default_path)$(savename).jld")
  catch loaderror
    println(loaderror)
    println("No $(default_path)$(savename).jld")
  end
  return features
end

function get_model_name(fm::FeatureMap,  kernelname::AbstractString,output_type::AbstractString, options)
    s=size(options.aux);
    name =string(fm.name,kernelname,"-",output_type)
    if(length(s)>1 || isempty(options.aux) )
        name = string(name,"-f");
    end
   return name
end
