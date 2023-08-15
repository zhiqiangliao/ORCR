using StatsBase

function normsq(n, d, SNR)

    x = rand(Uniform(-1, 1), n, d)
    y_t = sum((x[:,i]).^(2) for i in 1:d)

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function normsq_p(n, d, SNR)

    x = rand(Uniform(0, 1), n, d)
    y_t = sum((x[:,i]).^(2) for i in 1:d)

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function normsq_p2(n, d, SNR)

    x = rand(Uniform(0, 1), n, d)
    y_t = sum((x[:,i] .- 0.2).^(2) for i in 1:d)

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function normsq_p5(n, d, SNR)

    x = rand(Uniform(0, 1), n, d)
    y_t = sum((x[:,i] .- 0.5).^(2) for i in 1:d)

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function normsq_p7(n, d, SNR)

    x = rand(Uniform(0, 1), n, d)
    y_t = sum((x[:,i] .- 0.7).^(2) for i in 1:d)

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function normsq_p10(n, d, SNR)

    x = rand(Uniform(1, 10), n, d)
    y_t = 0.1/d .* sum((x[:,i]).^(2) for i in 1:d)

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function quadra(n, d, SNR; theta = 0.5)

    x = rand(Uniform(1, 10), n, d)
    Sigma = rand(Dirichlet(d, 1))
    Q = 0.1 .* Diagonal(Sigma)
    y_t = [x[i,:]'*Q*x[i,:] for i in 1:n]

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function quadra_unit(n, d, SNR; theta = 0.5)

    x = rand(Uniform(-1, 1), n, d)
    Q = Diagonal(ones(d)) .+ theta - theta* Diagonal(ones(d))
    y_t = [x[i,:]'*Q*x[i,:] for i in 1:n]

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    # standerization
    X = zeros(n,d);
    for ii=1:d
        X[:,ii] = x[:,ii] .- mean(x[:,ii])
        X[:,ii] = X[:,ii] ./ norm(x[:,ii])
    end
    y = y ./ norm(y)

    return X, y, y_t

end;

function norminf(n, d, SNR)

    x = rand(Uniform(-1, 1), n, d)
    y_t = sum(x[:,i].^(2) for i in 1:d) + 5 .*vec(maximum(abs.(x), dims=2))

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function production(n, d, SNR)

    x = rand(Uniform(1, 10), n, d)

    if d == 3
        y_t = 0.1 .* x[:,1] + 0.1 .* x[:,2] + 0.1 .* x[:,3] + 0.3 .* (x[:,2] .* x[:,2] .* x[:,3]).^(1/3)
    elseif d == 2
        y_t = 0.1 .* x[:,1] + 0.1 .* x[:,2] + 0.3 .* (x[:,1] .* x[:,2]).^(1/2)
    elseif d == 1
        y_t = log.(x) .+ 3 
    else
        println("error: Dimension exceeds 3.")
    end 
    
    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function linear(n, d, SNR)

    x = rand(Uniform(1, 10), n, d)
    y_t = 1/d .* x * ones(d)

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function Douglas(n, d, SNR)
    
    x = rand(Uniform(1, 10), n, d)
    y_t = [prod(x[i,:].^(0.8/d)) for i in 1:n]

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function Douglas_1(n, d, SNR)
    
    x = rand(Uniform(0, 1), n, d)
    y_t = [prod(x[i,:].^(0.8/d)) for i in 1:n]

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function s_shape(n, d, SNR)

    x = rand(Uniform(-1, 1), n)

    y_t = zeros(n)
    for i in 1:n
        if x[i] >=0
            y_t[i] = x[i]^0.88
        else
            y_t[i] = -2.25*(-x[i])^0.88
        end
    end

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return reshape(x,(n,d)), reshape(y,(n,d)), reshape(y_t,(n,d))

end;

function log_exp(n, d, SNR)
    
    x = rand(Uniform(1, 10), n, d)
    y_t = sum(exp.(0.2 .* x[:,i]) for i in 1:d)

    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;

function ill(n, d, SNR)

    x = rand(Uniform(1, 10), n, d)

    if d == 3
        y_t = 0.1 .* x[:,1] + 0.1 .* x[:,2] + 0.1 .* x[:,3] + 0.3 .* (x[:,2] .* x[:,3]).^(1/2)
    elseif d == 2
        y_t = 0.1 .* x[:,1] + 0.1 .* x[:,2] + 0.3 .* (x[:,1] .* x[:,2]).^(1/2)
    elseif d == 1
        y_t = log.(x) .+ 3 
    else
        println("error: Dimension exceeds 3.")
    end 
    
    sigma = sqrt(var(y_t) / SNR)
    epsilon = rand(Normal(0, sigma), n)
    y = y_t + epsilon

    return x, y, y_t

end;