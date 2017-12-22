# Takes an array of ResponseData and predicts the label based on the model in mlmodel 
function predict_linear(mom::StatisticalMoments,  fm::FeatureMap,  ml::ML_model, options)
    x = zeros(1,ml.nfeatures); 
    fm.apply(x, 1, mom,options);   # Applied the features map
    x = x_apply_transform(x,ml); #Applies any data transformation we applied to the training set including add the 1 bias term
    #Compute linear prediction implements representar theorem:  $<w^*, \psi(x )> = \sum_{\ell =1} (\alpha^*)^\ell K( x^\ell,x) $
    ys = ml.kernelapply(ml.X,x,ml.kernelparam)*ml.parameters;# println("ys before undo_transform: ",ys)
    #Undo the data transformations applied to y
    ys = y_undo_transform(ys,ml);
    return ys;
end
