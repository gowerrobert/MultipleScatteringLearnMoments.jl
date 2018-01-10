
## Here we show how to create the moments data, used by this package, from the simulations generated from [MultipleScattering.jl](https://github.com/jondea/MultipleScattering.jl). If you are unfamilar with Julia, please use  [JuliaBox](https://www.juliabox.com/).

First download the full simulated data to the current directory, this takes a while.
```julia
#training set
download("https://zenodo.org/record/1126642/files/bunny.mst.jld", "bunny.mst.jld")
#test set
download("https://zenodo.org/record/1126642/files/bunnytest.mst.jld", "bunnytest.mst.jld")
```
Get necessary packages to read the data.
```julia
Pkg.clone("https://github.com/jondea/MultipleScattering.jl.git")
using MultipleScattering

Pkg.add("JLD")
using JLD
```

Test set: create the moments of the backscattered waves in time
```julia
test = collect(values(load("bunnytest.mst.jld")))[1]
test = sort(test, by = model_to_label)
test_batched = collect(groupby(model_to_label, test));

tmp_labels = [model_to_label(bs[1]) for bs in test_batched];
if length(union(tmp_labels)) != length(tmp_labels)
  error("batched the test data inccorectly")
end

test_moments = map(test_batched) do bs
  time_arr = cut_interval(ω_to_t(bs[1].k_arr),[10.,96.])
  StatisticalMoments(
      [TimeSimulation(b;time_arr = time_arr, impulse = impulse) for b in bs],
      4;
      response_apply=real
  )
end;
```
Save the result, then clear the memory.
```julia
save("train.mnts.jld", "Array{StatisticalMoments}", test_moments)
test_batched = 0
test_moments = 0
gc() # garbage collection
```

Training set: create the moments of the backscattered waves in time.
```julia
  train = collect(values(load("bunny.mst.jld")))[1]
  train = sort(train, by = model_to_label)
  train_batched = collect(groupby(model_to_label, train));
  train = 0
  gc()

  tmp_labels = [model_to_label(bs[1]) for bs in train_batched];
  if length(union(tmp_labels)) != length(tmp_labels)
    error("batched the test data inccorectly")
  end

  train_moments = map(train_batched) do bs
    time_arr = cut_interval(ω_to_t(bs[1].k_arr),[10.,96.])
    StatisticalMoments(
        [TimeSimulation(b;time_arr = time_arr, impulse = impulse) for b in bs],
        4;
        response_apply=real
    )
  end;
  save("test.mnts.jld", "Array{StatisticalMoments}", train_moments)
  ```
