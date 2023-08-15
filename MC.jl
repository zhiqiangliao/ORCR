
function simulation_e1(n, d, SNR, seed; func="Prod")

    Random.seed!(seed)
    # generate train, validation and test data
    x, y, y_true = Douglas(2*n, d, SNR)
    x_tr, y_tr, y_true_tr = x[1:n,:], y[1:n], y_true[1:n]
    x_val, y_val, y_true_val = x[n:2*n,:], y[n:2*n], y_true[n:2*n]
    x_te, y_te, y_true_te = Douglas(1000, d, SNR)

    # tuning parameter on validation set
    p = [[0.1,0.5,0.8,1,2], # pcnls
        LinRange(0.1, 1, 10), # lcr
        LinRange(0.1, 1, 10), # alcr
        LinRange(0.05, 0.45, 9) # wrcr
        ]
    method = [pCNLS, LCR, ALCR, WRCR]
    para = zeros(length(method))
    @threads for (idx, i) in collect(enumerate(method))
        para[idx] = CV(i, x_val, y_val, y_true_val, p[idx]; func)
    end
    
    # solve the linear model
    alpha, beta_alcr = LIN(x_tr, y_tr)

    # solve the CNLS model
    alpha, beta_cnls = CNLS(x_tr, y_tr; func)
    y = alpha .+ sum(beta_cnls .* x_tr, dims = 2)
    mse_in_cnls = mean((y_true_tr - y).^2)
    mse_out_cnls = mean((yhat(alpha, beta_cnls, x_te; func) - y_true_te).^2)

    # solve the pCNLS model
    alpha, beta = pCNLS(x_tr, y_tr, para[1]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_pcnls = mean((y_true_tr - y).^2)
    mse_out_pcnls = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the LCR model
    alpha, beta = LCR(x_tr, y_tr, para[2]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_lcr = mean((y_true_tr - y).^2)
    mse_out_lcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the ALCR model
    alpha, beta = ALCR(x_tr, y_tr, para[3]; beta_0=beta_alcr, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_alcr = mean((y_true_tr - y).^2)
    mse_out_alcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the WRCR model
    alpha, beta = WRCR(x_tr, y_tr, para[4]; beta_1=beta_cnls, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_wrcr = mean((y_true_tr - y).^2)
    mse_out_wrcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    return mse_in_cnls, mse_in_pcnls, mse_in_lcr, mse_in_alcr, mse_in_wrcr, mse_out_cnls, mse_out_pcnls, mse_out_lcr, mse_out_alcr, mse_out_wrcr
    
end;

function simulation_e2(n, d, SNR, seed; func="Convex")

    Random.seed!(seed)
    # generate train, validation and test data
    x, y, y_true = normsq_2(2*n, d, SNR)
    x_tr, y_tr, y_true_tr = x[1:n,:], y[1:n], y_true[1:n]
    x_val, y_val, y_true_val = x[n:2*n,:], y[n:2*n], y_true[n:2*n]
    x_te, y_te, y_true_te = normsq_2(1000, d, SNR)

    # tuning parameter on validation set
    p = [[0.001,0.01,0.1,1], # pcnls
        LinRange(0.5, 3, 6), # lcr
        LinRange(0.5, 3, 6), # alcr
        LinRange(0.05, 0.45, 9) # wrcr
        ]
    method = [pCNLS, LCR, ALCR, WRCR]
    para = zeros(length(method))
    @threads for (idx, i) in collect(enumerate(method))
        para[idx] = CV(i, x_val, y_val, y_true_val, p[idx]; func)
    end
    
    # solve the linear model
    alpha, beta_alcr = LIN(x_tr, y_tr)

    # solve the CNLS model
    alpha, beta_cnls = CNLS(x_tr, y_tr; func)
    y = alpha .+ sum(beta_cnls .* x_tr, dims = 2)
    mse_in_cnls = mean((y_true_tr - y).^2)
    mse_out_cnls = mean((yhat(alpha, beta_cnls, x_te; func) - y_true_te).^2)

    # solve the pCNLS model
    alpha, beta = pCNLS(x_tr, y_tr, para[1]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_pcnls = mean((y_true_tr - y).^2)
    mse_out_pcnls = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the LCR model
    alpha, beta = LCR(x_tr, y_tr, para[2]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_lcr = mean((y_true_tr - y).^2)
    mse_out_lcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the ALCR model
    alpha, beta = ALCR(x_tr, y_tr, para[3]; beta_alcr, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_alcr = mean((y_true_tr - y).^2)
    mse_out_alcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the WRCR model
    alpha, beta = WRCR(x_tr, y_tr, para[4]; beta_1=beta_cnls, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_wrcr = mean((y_true_tr - y).^2)
    mse_out_wrcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    return mse_in_cnls, mse_in_pcnls, mse_in_lcr, mse_in_alcr, mse_in_wrcr, mse_out_cnls, mse_out_pcnls, mse_out_lcr, mse_out_alcr, mse_out_wrcr
    
end;

function simulation_norm(n, d, SNR, seed; func="Convex")

    Random.seed!(seed)
    # generate train, validation and test data
    x, y, y_true = normsq_p2(2*n, d, SNR)
    x_tr, y_tr, y_true_tr = x[1:n,:], y[1:n], y_true[1:n]
    x_val, y_val, y_true_val = x[n:2*n,:], y[n:2*n], y_true[n:2*n]
    x_te, y_te, y_true_te = normsq_p2(1000, d, SNR)

    # tuning parameter on validation set
    p = [[0.001,0.01,0.1,1], # pcnls
        LinRange(0.5, 3, 10), # lcr
        LinRange(0.5, 3, 10), # alcr
        LinRange(0.05, 0.45, 9) # wrcr
        ]
    method = [pCNLS, LCR, ALCR, WRCR]
    para = zeros(length(method))
    @threads for (idx, i) in collect(enumerate(method))
        para[idx] = CV(i, x_val, y_val, y_true_val, p[idx]; func)
    end
    
    # solve the linear model
    alpha, beta_alcr = LIN(x_tr, y_tr)

    # solve the CNLS model
    alpha, beta_cnls = CNLS(x_tr, y_tr; func)
    y = alpha .+ sum(beta_cnls .* x_tr, dims = 2)
    mse_in_cnls = mean((y_true_tr - y).^2)
    mse_out_cnls = mean((yhat(alpha, beta_cnls, x_te; func) - y_true_te).^2)

    # solve the pCNLS model
    alpha, beta = pCNLS(x_tr, y_tr, para[1]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_pcnls = mean((y_true_tr - y).^2)
    mse_out_pcnls = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the LCR model
    alpha, beta = LCR(x_tr, y_tr, para[2]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_lcr = mean((y_true_tr - y).^2)
    mse_out_lcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the ALCR model
    alpha, beta = ALCR(x_tr, y_tr, para[3]; beta_0=beta_alcr, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_alcr = mean((y_true_tr - y).^2)
    mse_out_alcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the WRCR model
    alpha, beta = WRCR(x_tr, y_tr, para[4]; beta_1=beta_cnls, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_wrcr = mean((y_true_tr - y).^2)
    mse_out_wrcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    return mse_in_cnls, mse_in_pcnls, mse_in_lcr, mse_in_alcr, mse_in_wrcr, mse_out_cnls, mse_out_pcnls, mse_out_lcr, mse_out_alcr, mse_out_wrcr
    
end;

function simulation_ill(n, d, SNR, seed; func="Prod")

    Random.seed!(seed)
    # generate train, validation and test data
    x, y, y_true = production(2*n, d, SNR)
    x_tr, y_tr, y_true_tr = x[1:n,:], y[1:n], y_true[1:n]
    x_val, y_val, y_true_val = x[n:2*n,:], y[n:2*n], y_true[n:2*n]
    x_te, y_te, y_true_te = production(1000, d, SNR)

    # tuning parameter on validation set
    p = [[0.1,1,2,3,4,5,6,7,8,10], # pcnls
        [0.1,0.2,0.3,0.4,0.5,0.8], # lcr
        [0.001,0.01,0.08,0.1,0.15,0.2,0.5], # alcr
        # [0.01,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45] # wrcr
        [0.001,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8] # awrcr
        ]
    method = [pCNLS, LCR, ALCR, WRCR]
    para = zeros(length(method))
    @threads for (idx, i) in collect(enumerate(method))
        para[idx] = CV(i, x_val, y_val, y_true_val, p[idx]; func)
    end
    
    # solve the linear model
    alpha, beta_alcr = LIN(x_tr, y_tr)
    y_linear = alpha .+ x_tr * beta_alcr
    mse_in_lin = mean((y_true_tr - y_linear).^2)
    mse_out_lin = mean((y_true_te - (alpha .+ x_te * beta_alcr) ).^2)

    # solve the CNLS model
    alpha, beta_cnls = CNLS(x_tr, y_tr; func)
    y = alpha .+ sum(beta_cnls .* x_tr, dims = 2)
    mse_in_cnls = mean((y_true_tr - y).^2)
    mse_out_cnls = mean((yhat(alpha, beta_cnls, x_te; func) - y_true_te).^2)
    max_cnls = maximum(beta_cnls, dims=1)
    min_cnls = minimum(beta_cnls, dims=1)
    var_cnls = var(beta_cnls, dims=1)

    # solve the pCNLS model
    alpha, beta = pCNLS(x_tr, y_tr, para[1]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_pcnls = mean((y_true_tr - y).^2)
    mse_out_pcnls = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)
    max_pcnls = maximum(beta, dims=1)
    min_pcnls = minimum(beta, dims=1)
    var_pcnls = var(beta, dims=1)

    # solve the LCR model
    alpha, beta = LCR(x_tr, y_tr, para[2]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_lcr = mean((y_true_tr - y).^2)
    mse_out_lcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)
    max_lcr = maximum(beta, dims=1)
    min_lcr = minimum(beta, dims=1)
    var_lcr = var(beta, dims=1)

    # solve the ALCR model
    alpha, beta = ALCR(x_tr, y_tr, para[3]; beta_0=beta_alcr, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_alcr = mean((y_true_tr - y).^2)
    mse_out_alcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)
    max_alcr = maximum(beta, dims=1)
    min_alcr = minimum(beta, dims=1)
    var_alcr = var(beta, dims=1)

    # solve the WRCR model
    alpha, beta = AWRCR(x_tr, y_tr, para[4]; beta_0=beta_alcr, beta_1=beta_cnls, func)
    # alpha, beta = WRCR(x_tr, y_tr, para[4]; beta_1=beta_cnls, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_wrcr = mean((y_true_tr - y).^2)
    mse_out_wrcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)
    max_wrcr = maximum(beta, dims=1)
    min_wrcr = minimum(beta, dims=1)
    var_wrcr = var(beta, dims=1)

    out = [mse_out_lin, mse_out_cnls, mse_out_pcnls, mse_out_lcr, mse_out_alcr, mse_out_wrcr]
    in = [mse_in_lin, mse_in_cnls, mse_in_pcnls, mse_in_lcr, mse_in_alcr, mse_in_wrcr]
    beta_max = hcat(reshape(beta_alcr, (1,3)), max_cnls, max_pcnls, max_lcr, max_alcr, max_wrcr)
    beta_min = hcat(reshape(beta_alcr, (1,3)), min_cnls, min_pcnls, min_lcr, min_alcr, min_wrcr)
    beta_sd = hcat(var_cnls, var_pcnls, var_lcr, var_alcr, var_wrcr)

    return in, out, beta_max, beta_min, beta_sd
    
end;

function simulation_miss(n, d, SNR, seed; func="Concave", DGP = Douglas)

    Random.seed!(seed)
    # generate train, validation and test data
    x, y, y_true = DGP(2*n, d, SNR)
    x_tr, y_tr, y_true_tr = x[1:n,:], y[1:n], y_true[1:n]
    x_val, y_val, y_true_val = x[n:2*n,:], y[n:2*n], y_true[n:2*n]
    x_te, y_te, y_true_te = DGP(1000, d, SNR)

    # tuning parameter on validation set
    p = [[0.001,0.01,0.1,1,1.5], # pcnls
        [0.1,0.4,0.6,0.8,1], # lcr
        [0.0001,0.1,0.4,0.6,0.8,1], # alcr
        [0.1,0.2,0.3,0.4,0.45] # wrcr
        ]
    method = [pCNLS, LCR, ALCR, WRCR]
    para = zeros(length(method))
    @threads for (idx, i) in collect(enumerate(method))
        para[idx] = CV(i, x_val, y_val, y_true_val, p[idx]; func)
    end
    
    # solve the linear model
    alpha, beta_alcr = LIN(x_tr, y_tr)
    y_linear = alpha .+ x_tr * beta_alcr
    mse_in_linear = mean((y_true_tr - y_linear).^2)
    mse_out_linear = mean((y_true_te - (alpha .+ x_te * beta_alcr) ).^2)

    # solve the CNLS model
    alpha, beta_cnls = CNLS(x_tr, y_tr; func)
    y = alpha .+ sum(beta_cnls .* x_tr, dims = 2)
    mse_in_cnls = mean((y_true_tr - y).^2)
    mse_out_cnls = mean((yhat(alpha, beta_cnls, x_te; func) - y_true_te).^2)

    # solve the pCNLS model
    alpha, beta = pCNLS(x_tr, y_tr, para[1]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_pcnls = mean((y_true_tr - y).^2)
    mse_out_pcnls = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the LCR model
    alpha, beta = LCR(x_tr, y_tr, para[2]; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_lcr = mean((y_true_tr - y).^2)
    mse_out_lcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the ALCR model
    alpha, beta = ALCR(x_tr, y_tr, para[3]; beta_0=beta_alcr, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_alcr = mean((y_true_tr - y).^2)
    mse_out_alcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    # solve the WRCR model
    alpha, beta = WRCR(x_tr, y_tr, para[4]; beta_1=beta_cnls, func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_wrcr = mean((y_true_tr - y).^2)
    mse_out_wrcr = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    return mse_in_cnls, mse_in_pcnls, mse_in_lcr, mse_in_alcr, mse_in_wrcr, mse_in_linear, mse_out_cnls, mse_out_pcnls, mse_out_lcr, mse_out_alcr, mse_out_wrcr, mse_out_linear
    
end;

function simulation_missp(n, d, SNR, seed; func="Concave", DGP = Douglas)

    Random.seed!(seed)
    # generate train, validation and test data
    x, y, y_true = DGP(2*n, d, SNR)
    x_tr, y_tr, y_true_tr = x[1:n,:], y[1:n], y_true[1:n]
    x_val, y_val, y_true_val = x[n:2*n,:], y[n:2*n], y_true[n:2*n]
    x_te, y_te, y_true_te = DGP(1000, d, SNR)

    # tuning parameter on validation set
    p = [1,10,50,80,100,150,200] # pcnls
        
    para = CV(pCNLS, x_val, y_val, y_true_val, p; func)
    
    # solve the pCNLS model
    alpha, beta = pCNLS(x_tr, y_tr, para; func)
    y = alpha .+ sum(beta .* x_tr, dims = 2)
    mse_in_pcnls = mean((y_true_tr - y).^2)
    mse_out_pcnls = mean((yhat(alpha, beta, x_te; func) - y_true_te).^2)

    return mse_in_pcnls, mse_out_pcnls
    
end;