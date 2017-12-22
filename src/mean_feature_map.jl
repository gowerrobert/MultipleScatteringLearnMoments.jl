## Functions for mean feature map
function apply_mean(X::Array{Float64}, rowcounts::Int64, mom::StatisticalMoments,options)
    # X is the matrix formed by all data inputs. All "apply" functions for feature maps need to fill in X  [rowcount,:] with the mapped features.
   lenresponse = length(mom.moments[1]);
   X[rowcounts,1:lenresponse] = mom.moments[1];
   # NOTE: scientifically it's important to see how well mean(real.(responses),2) performs, and not to use abs. If someone could measure the abs of each response, then they could use any statistic they wanted really. However, many experiments (and theoretical developments) use only mean(real.(responses),2).   #Works better with abs.(responses)
end


function get_num_mean_features(mom::StatisticalMoments, options)
   lenresponse = length(mom.moments[1]);
    return lenresponse;
end


# Defines the feature map
fmmean = FeatureMap(apply_mean,get_num_mean_features,"meanmap");
##end functions for identity feature map.
