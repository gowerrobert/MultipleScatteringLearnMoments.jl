# Definitions and functions for defining different Kernels
#type KernelLinear

#end
##Retrieves the kernel matrix function and kernal element function given its name
function get_kernel_function(kernelname::AbstractString, kernelparam::Float64)
    if(kernelname == "gauss")
        #println("Using the Gaussian Kernel-",kernelparam)
        return X -> applykmatrix(X,apply_gaussk,kernelparam), apply_gaussk;
    elseif(kernelname == "OrnUhlen")
       # println("Using the Ornstein–Uhlenbeck Kernel-",kernelparam)
        return X -> applykmatrix(X,apply_OrnUhlenk,kernelparam), apply_OrnUhlenk;
    elseif( kernelname =="rationquad")
        #println("Using the Rational quadratic Kernel-",kernelparam)
        return X -> applykmatrix(X,apply_rationquadk,kernelparam), apply_rationquadk;
    else
        return identity, apply_identity;
    end
end

function apply_identity(X::Array{Float64}, x, kernelparam::Float64)
    return x;
end


function apply_gaussk(X::Array{Float64}, x,kernelparam::Float64)
    sX  =size(X);
    gx = zeros(1,sX[1])
    for i =1:sX[1]
        gx[i] = exp(-(norm(x - X[i,:]').^2)./(kernelparam.^2));
    end
    return gx;
end

function apply_OrnUhlenk(X::Array{Float64}, x,kernelparam::Float64) # Ornstein–Uhlenbeck
    sX  =size(X);
    gx = zeros(1,sX[1])
    for i =1:sX[1]
        gx[i] = exp(-(norm(x - X[i,:]')/kernelparam));
    end
    return gx;
end


function apply_rationquadk(X::Array{Float64}, x,kernelparam::Float64) # Rational quadratic
    sX  =size(X);
    gx = zeros(1,sX[1])
    for i =1:sX[1]
        gx[i] =(1+norm(x - X[i,:]')^1.0).^(-kernelparam);
    end
    return gx;
end

function applykmatrix(X::Array{Float64}, applyfunc::Function,kernelparam::Float64)
    sX  =size(X);
    gX = zeros(sX[1],sX[1]); # The kernel matrix has as many rows and columns as there are data points
    for i =1:sX[1]
        gX[i,:] =  applyfunc(X,X[i,:]',kernelparam)
    end
    return gX;
end


##apply_kernel_whole_matrix

##apply_kernel_test_data