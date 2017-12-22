#Calculate the data transformations to the output
function y_transform(ys::Array{Float64},ml::ML_model) 
    ml.yscale  = mean(ys); ys[:]=ys./ ml.yscale ; ys[:] = log.(ys);     # Scaling and log transform
 #   ml.yscale = std(ys,1); ml.ysmean  =mean(ys,1); ys[:]= (ys.-ml.ysmean)./ml.yscale; # Centering and scaling
    return ys
end

#Undo the data transformations to the output
function y_undo_transform(ys::Array{Float64}, ml::ML_model) 
    ys[:] = exp.(ys); ys[:] = ys.*ml.yscale;    # Scaling and log transform
  #  ys[:]= ys.*ml.yscale.+ml.ysmean;     # Centering and scaling
    return ys
end

#Calculate the data transformations to the input x (feature vector)
function X_transform(X::Array{Float64}, ml::ML_model) 
    ml.Xmean = mean(X,1);
  #  ml.Xstd = std(X,1);  
  #  ind = (0.==ml.Xstd); ml.Xstd[ind] =1.0;  #replace 0 in std by 1 incase there is a constant feature
  #  X= (X.-ml.Xmean)./ml.Xstd; # Centering and scaling
    X= X.-ml.Xmean;
    sX = size(X);
    X = [X ones(sX[1],1)]; # Adding the bias term
    return X
end

#Apply the saved data transformations to a single data vector x (feature vector)
function x_apply_transform(x::Array{Float64}, ml::ML_model) 
  #  x= (x.-ml.Xmean)./ml.Xstd; # Centering and scaling
    x =x.-ml.Xmean;
    x = [x 1];# Adding the bias term
   return x
end