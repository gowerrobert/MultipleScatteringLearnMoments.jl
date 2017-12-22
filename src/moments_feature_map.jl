## Functions for centered moments feature maps
#  m1limextra=0;
#  m2limextra=0;
function apply_moments(X::Array{Float64}, rowcounts::Int64, mom::StatisticalMoments,options)
   lenresponse = length(mom.moments[1])
    momnums = options.aux;
    s=size(momnums);
    count =0; i=1;
    if(length(s)==1)
        for i in momnums
            X[rowcounts,count*lenresponse+1:(count+1)*lenresponse ]  = mom.moments[i];
            count = count+1;
        end
    else
        for mm in mom.moments
            mask =options.aux[i,:] .>0 
            mmmasked =  mm[mask];
            lengthmask = sum(options.aux[i,:]);
            X[rowcounts,count+1:count+lengthmask] =  mmmasked;
            count = count +lengthmask;
            i = i+1;
        end
        ##
    end
end

function get_num_moments_features(mom::StatisticalMoments, options)
    s=size(options.aux);
    if(length(s)==1)
        lenresponse = length(options.aux)*length(mom.moments[1]);#+(m2limextra)*(m1limextra); 
    else
        lenresponse = sum(options.aux);
    end
    return lenresponse;
end

# Defines the feature map
fmmoments = FeatureMap(apply_moments,get_num_moments_features, "momentsmap");
##end functions for identity feature map.
