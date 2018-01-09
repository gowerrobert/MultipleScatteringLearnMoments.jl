using MultipleScattering
using JLD

model_to_label(f) = (f.volfrac,f.particles[1].r)
impulse = get_gaussian_freq_impulse(1.0,2.48)

# Download data, can take a long time.
  #training set
  download("https://zenodo.org/record/1126642/files/bunny.mst.jld", "bunny.mst.jld")
  #test set
  download("https://zenodo.org/record/1126642/files/bunnytest.mst.jld", "bunnytest.mst.jld")

# test set: create the moments of the backscattered waves in time
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
  save("train.mnts.jld", "Array{StatisticalMoments}", test_moments)

  test_batched = 0
  test_moments = 0
  gc() # garbage collection

# training set: create the moments of the backscattered waves in time
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
