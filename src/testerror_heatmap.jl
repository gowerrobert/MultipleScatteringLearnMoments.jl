
function testerror_heatmap(test::Array{StatisticalMoments{Float64}},fm::FeatureMap,  ml::ML_model,
      options, output_type::AbstractString;
      strdir="default")

     pairs = unique([ td.label[1]/(td.label[2]^2), td.label[2]] for td in test);
    #pairs = unique([ td.volfrac/(td.particles[1].r^2), td.particles[1].r] for td in tds);
    zerohypo = mean(pairs.^2);
    Ferrvol = mom -> Ferr_internal(mom, ml,fm,zerohypo,options, output_type)
    plotmom_heatmap(test,Ferrvol)
    if strdir == "default"
      strdir = isdir("../figures/") ? "../figures" : ""
    end
    savefig("$(strdir)heatmap-$(ml.name).pdf");
end

function Ferr_internal(mom::StatisticalMoments, ml::ML_model,fm::FeatureMap, zerohypo::Array{Float64},options, output_type::AbstractString)
    ypred = predict_linear(mom, fm, ml, options); ypred= ypred[1];
    rd = mom.label[2];
    if(output_type =="radius")
        ferror =  (ypred-rd)^2/zerohypo[2];
    else
        vl  =mom.label[1] ;
        vlf = vl/rd^2;
        ferror = (ypred-vlf)^2/zerohypo[1];
     end# +abs(ypred[2] -rd)/rd;   abs(ypred[1]-vlf)/vlf
    return ferror;

end
    
    function plotmom_heatmap(moms::Array{StatisticalMoments{Float64}}, contract_tds = mom ->  mom.label[2] )
  label_f = mom -> [mom.label[1],mom.label[2]]
  labels = union([ label_f(td)  for td in moms])
  #tds_arr = [filter(td -> label_f(td) == l, moms) for l in labels]
  F_arr = [contract_tds(tds) for tds in moms];
  vols = sort([l[1] for l in labels])
  as = sort([l[2] for l in labels])
  z_matrix = NaN*zeros(length(as),length(vols))
  for i in indices(F_arr,1)
    n = find(labels[i][1] .== vols)
    m = find(labels[i][2] .== as)
    z_matrix[m,n] = F_arr[i]
  end
  heatmap(vols,as,z_matrix, xlabel="numfrac", ylabel="radius")
end
        
        
