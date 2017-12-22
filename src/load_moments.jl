#function for fitting a linear regression. Takes as input
#  tds::Array{ResponseData}      Training data set
#  fm::FeatureMap                Feature map  as defined in ML_models.jl
#  options                       Contains the regularization parameters and other optional arguments
# NOTE: You can not use a kernal combined with an L1 regularizor since we can no longer use the "kernel trick". But we can still use  feature maps.
function findfolder(strname)
  strs= ["/../../$strname","/../$strname","/$strname" ]
  strfolder = filter(isdir, map(str -> string(pwd(),str),strs))
  if length(strfolder) == 0
    return ""
  elseif length(strfolder) > 1
    warn("found more than on $strname folder")
  end
  return strfolder[1]
end

function findfiles(strdata)
  strbase = basename(strdata)
  strdir = dirname(strdata)
  strdir =  (strdir=="") ? "" : string("/",strdir)
  # will look for data in three folders:
  folders =[string(pwd(),"/..",strdir),string(pwd(),strdir)]
  if strdir != "/data" && strdir != "/Data"
    push!(folders, findfolder("data"))
    push!(folders, findfolder("Data"))
  end
  filter!(f -> isdir(f), folders)
  files = vcat(map(str -> map(s -> string(str,"/",s), readdir(str)), folders)...)
  filter!(f -> contains(f, strbase), files)
end

function jlsTojld(strdata)
  if ((str =split(strdata,'.')[end]) != "jls")
    error("can only convert jls, not type $str, to jld")
  end
  rds = load_rds(strdata)
  rd_pairs = [
    ("(volfrac,a)=($(rd.volfrac),$(rd.particles[1].r))",rd)
  for rd in rds]
  strfile=string(split(strdata,'.')[1],".jld")
  save(strfile, Dict(rd_pairs))
  return Dict(rd_pairs)
end

function load_moments(strfile)
  if !isfile(strfile)
    strfile =  filter(s -> split(s,'.')[length(split(s,'.'))-1]=="mnts" ,findfiles(strfile))[1]
  end
  println("loading:", strfile)
  strs = split(strfile,'.')
  if strs[end] == "jls"
    moments = open(strfile, "r") do f deserialize(f) end
  elseif strs[end] == "jld"
    moments = Array{StatisticalMoments{Float64}}(collect(values(load(strfile)))[1])
  else error("file type not recognised")
  end
  return moments
end