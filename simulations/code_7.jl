using Statistics, Random, LinearAlgebra, CSV, DataFrames, Base.Threads, Distributions
include("../DGP.jl")
include("../MC.jl")
include("../estimator.jl")
include("../toolbox.jl")

M = 50
n = 500
d = 3
SNR = 3

# simulation
error = zeros(M,10)
@threads for i in 1:M
    error[i,:] .= simulation_e1(n, d, SNR, i)
end

df = DataFrame(error,["in_cnls", "in_pcnls", "in_lcr", "in_alcr", "in_wrcr", 
                        "out_cnls", "out_pcnls", "out_lcr", "out_alcr", "out_wrcr"])

CSV.write("mse$(n)_$(d)_$(SNR).csv", df)
