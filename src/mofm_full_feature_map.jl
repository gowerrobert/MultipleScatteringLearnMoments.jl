## Functions for centered moments of moments feature maps
#const m1s2 = [1,2,3,4,5];
const m2lim2 = 10;
function apply_mofm_full(X::Array{Float64}, rowcounts::Int64, mom::StatisticalMoments,options)
   lenresponse = length(mom.moments[1]);
   momnums = options.aux;
   s=size(momnums);
   count =1; i =1;
    if(length(s)==1)
        for i in momnums
            mm  = mom.moments[i];
            for j = 1:m2lim2
                X[rowcounts,count] = get_moment_vec(mm,j);
                count = count +1;
            end
        end
   else
        for mm in mom.moments
            for j = 1:m2lim2
                if(options.aux[i,j]==1)
                   X[rowcounts,count] = get_moment_vec(mm,j);
                   count = count +1;
                end
            end
            i = i +1;
        end 
   end  
end

function get_num_mofm__full_features(mom::StatisticalMoments, options)
    s=size(options.aux);
    if(length(s)==1)
        lenresponse =  length(options.aux)*m2lim2;
     else 
        lenresponse = sum(options.aux);
    end
    return lenresponse;
end

# Defines the feature map
fmofm_full = FeatureMap(apply_mofm_full,get_num_mofm__full_features, "mofm_full_map");

#"returns the  mom_num-th moment of resps, where resps is a vector whose elements can be either numbers or vectors."
function get_moment_vec(resps, mom_num)
    mu = mean(resps);
    if(mom_num==1) return mu end;
    num_sim = length(resps);
    if(num_sim<2)
        println("Not enough simulations to calculate momments, return zeros")
        return 0.0*mu
    end
    moment = sum((mu - r).^mom_num for r in resps)./(num_sim-1.0)
    return sign(moment).*abs(moment).^(1.0/mom_num) # I think the sign does make a difference, as it changes with different labels
end

