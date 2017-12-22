#Calculate the test error of prediction using the model ml on the test data set
function test_linear(test::Array{StatisticalMoments{Float64}}, testname::AbstractString, fm::FeatureMap,  ml::ML_model, options, output_type::AbstractString;
  title="default", strdir="default") #km_transform::Function,
  # Get all volfracs and voliuses
  volfracs= union([td.label[1] for td in test]);
  radiuses = union([td.label[2] for td in test]);
  concentrations = union([td.label[1]/(td.label[2]^2) for td in test]);
  testerror =0;  Rsqd =0;  meanerror = 0;  zerohypo =0;
  testerrors = zeros(length(volfracs)*length(radiuses));
  #volfracs = sort([vl for vl in volfracs]);
  #radiuses = sort([rd for rd in radiuses]);
  if(output_type =="radius")
    meanhypo = mean(radiuses);
  elseif(output_type =="volumefraction")
    meanhypo = mean(volfracs); #volume fraction        
  else
    meanhypo = mean(concentrations); #concentration
  end

  if(output_type =="radius")
    getdt = mom ->  mom.label[2];  # radius
  elseif(output_type =="volumefraction")
    getdt = mom ->  mom.label[1]; #volume fraction
  else
    getdt = mom ->  mom.label[1]/(mom.label[2]^2);  # concentration
  end
  reptd = collect([getdt(mom) for mom in test]);
  meanerror = sum((meanhypo-reptd).^2);
  zerohypo = sum(reptd.^2);
  pred = collect([ predict_linear(mom, fm, ml, options)[1] for mom in test ]);
  squarederror = sum((pred.-reptd).^2);
  absoluterror = sqrt(squarederror) /sqrt(zerohypo);
  Rsqd = 1 - squarederror/meanerror;

  if(options.plotfigures)
    println(output_type," ",floor(1000.0*absoluterror)/10.0, "% absolute test error")
    ## Plotting the regression
    #println("Plotting...check folder ./figures for the plots scatvol-$(ml.name)")
    try gr() catch pyplot() end # if gr() not available use pyplot()
    #pgfplots();
    gap = maximum(reptd)-minimum(reptd);
    if title == "default" title = "Rsqd = $(round(100*Rsqd)/100)" end
    scatter(reptd,pred,  xlabel = "true $(output_type)",
    ylabel = "predicted $(output_type)", label = "",  title=title)
    line  =minimum(reptd):gap/100.01:maximum(reptd);
    plot!(line,line,   label = "")
    if strdir == "default"
      strdir = isdir("../figures/") ? "../figures/" : ""
    end
     open("./test_output.txt", "w") do f
        write(f, "$(output_type)\n")
            for td in reptd
               write(f, "$(td)\n") 
            end
        write(f, "\n\n")     
        write(f, "predicted\n")
            for td in pred
               write(f, "$(td)\n") 
            end  
        write(f, "\n\n")              
        write(f, "line\n")
            for td in line
               write(f, "$(td)\n") 
            end                         
     end  
#     savefig("$(strdir)scat-$(ml.name)-$(testname).svg");
    savefig("$(strdir)scat-$(ml.name)-$(testname).pdf");
    println("R squared error $(output_type)= ",round(100*Rsqd)/100) #round(100*Rsqd)/100
    gr()
    testerror_heatmap(test, fm, ml, options, output_type; strdir=strdir)
  end

  return round(100*Rsqd)/100
end


## OLD version

#    rep =[]; #The repeated test outputs
#     pred = []; #The predicted outputs
#     totcount =1;
#  #     label_f = mom -> [mom.label[1],mom.label[2]]
#   #labels = union([ label_f(td)  for td in moms])

#     for mom in test
#             vl = mom.label[1] ;
#             rd = mom.label[2] ;
#             if(output_type =="radius")
#                 dt = rd;
#             else
#                 dt = vl/(rd)^2;
#             end
#             rep = [rep ; dt];
#             ypred = predict_linear(mom, fm, ml, options);
#             ypred= ypred[1];
#             pred = [pred ; ypred];
#             testerrors[totcount] =  abs(ypred -dt)/dt;
#             Rsqd = Rsqd+ (ypred -dt)^2; # Accumulating R squared error
#             meanerror= meanerror+ (meanhypo-dt)^2;#Calculating the error of the  meanerrorthesis
#             zerohypo= zerohypo+ dt^2;#Calculating the error of the  meanerrorthesis
#             if(isnan(testerror))
#                 print("dt: ", dt, "ypred: ",ypred)
#             end
#             totcount = totcount+1;
# #        end
#     end
#     testerror  = mean(testerrors);
#     absoluterror = sqrt(Rsqd) /sqrt(zerohypo);
#     println("squared error: ", Rsqd)
#     Rsqd = 1- Rsqd/meanerror;
