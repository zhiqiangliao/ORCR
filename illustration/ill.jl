using Statistics, Random, LinearAlgebra, CSV, DataFrames, Base.Threads, Distributions
include("../DGP.jl")
include("../MC.jl")
include("../estimator.jl")
include("../toolbox.jl")

M = 50
n = 100
d = 3
SNR = 3

# simulation
error_in = zeros(M,6)
error_out = zeros(M,6)
beta_max = zeros(M, 18)
beta_min = zeros(M, 18)
beta_sd = zeros(M, 15)
@threads for i in 1:M
    res = simulation_ill(n, d, SNR, i)
    error_in[i,:] .= res[1]
    error_out[i,:] .= res[2]
    beta_max[i,:] = res[3]
    beta_min[i,:] = res[4]
    beta_sd[i,:] = res[5]

end

df_in = DataFrame(error_in,["in_linear", "in_cnls", "in_pcnls", "in_lcr", "in_alcr", "in_wrcr"])

df_out = DataFrame(error_out,["out_linear", "out_cnls", "out_pcnls", "out_lcr", "out_alcr", "out_wrcr"])

df_max = DataFrame(beta_max, :auto)

df_min = DataFrame(beta_min, :auto)

df_sd = DataFrame(beta_sd, :auto)

CSV.write("error_in.csv", df_in)
CSV.write("error_out.csv", df_out)
CSV.write("beta_max.csv", df_max)
CSV.write("beta_min.csv", df_min)
CSV.write("beta_sd.csv", df_sd)
