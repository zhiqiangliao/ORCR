using ScikitLearn

# Calculate yhat in testing sample
function yhat(alpha::Vector, beta::Matrix, x_test::Matrix; func::String="Convex")

    if x_test isa Array{<:Number,1}
        if func == "Prod"
            yhat = minimum(alpha + beta * x_test)
        elseif func == "Concave"
            yhat = minimum(alpha + beta * x_test)
        else
            yhat = maximum(alpha + beta * x_test)
        end
    else
        n = size(x_test)[1]
        yhat = zeros(n,)
        for i in 1:n
            if func == "Prod"
                yhat[i] = minimum(alpha + beta * x_test[i,:])
            elseif func == "Concave"
                yhat[i] = minimum(alpha + beta * x_test[i,:])
            else
                yhat[i] = maximum(alpha + beta * x_test[i,:])
            end
        end
    end

    return yhat
    
end;

function CV(estimator, x, y, y_true, C; n_fold = 5, shuffle=true, func="Convex")
    # compute the MSE for selecting the turning parameter C
    n = size(x)[1]

    # resample the data
    kfold = CrossValidation.KFold(n, n_folds=n_fold, random_state=1, shuffle=shuffle)

    error = zeros(length(C), n_fold)
    @threads for kk in 1:n_fold
        x_train, y_train, x_test, y_test = x[kfold[kk][1], :], y[kfold[kk][1]], x[kfold[kk][2], :], y_true[kfold[kk][2]]

        # estimate the model
        error_tmp = zeros(length(C))
        for (idx, i) in enumerate(C)
            if estimator == ALCR 
                alpha, beta_0 = LIN(x_train, y_train)
                alpha, beta = estimator(x_train, y_train, i; beta_0=beta_0, func)
            elseif estimator == WRCR
                alpha, beta_1 = CNLS(x_train, y_train; func)
                alpha, beta = estimator(x_train, y_train, i; beta_1=beta_1, func)
            elseif estimator == AWRCR
                alpha, beta_0 = LIN(x_train, y_train)
                alpha, beta_1 = CNLS(x_train, y_train; func)
                alpha, beta = estimator(x_train, y_train, i; beta_0=beta_0, beta_1=beta_1, func)
            else 
                alpha, beta = estimator(x_train, y_train, i; func)
            end
            mse = mean((yhat(alpha, beta, x_test; func) - y_test).^2)
            error_tmp[idx] = mse
        end
    
        error[:, kk] = error_tmp
    
    end
    pos = argmin(mean(error, dims=2))

    return C[pos]

end;

function CV_real(estimator, x, y, C; n_fold = 5, shuffle=true, func="Convex")
    # compute the MSE for selecting the turning parameter C
    n = size(x)[1]

    # resample the data
    kfold = CrossValidation.KFold(n, n_folds=n_fold, random_state=1, shuffle=shuffle)

    error = zeros(length(C), n_fold)
    @threads for kk in 1:n_fold
        x_train, y_train, x_test, y_test = x[kfold[kk][1], :], y[kfold[kk][1]], x[kfold[kk][2], :], y[kfold[kk][2]]

        # estimate the model
        error_tmp = zeros(length(C))
        for (idx, i) in enumerate(C)
            if estimator == ALCR 
                alpha, beta_0 = LIN(x_train, y_train)
                alpha, beta = estimator(x_train, y_train, i; beta_0=beta_0, func)
            elseif estimator == WRCR
                alpha, beta_0 = CNLS(x_train, y_train; func)
                alpha, beta = estimator(x_train, y_train, i; beta_0=beta_0, func)
            else 
                alpha, beta = estimator(x_train, y_train, i; func)
            end
            mse = mean((yhat(alpha, beta, x_test; func) - y_test).^2)
            error_tmp[idx] = mse
        end
    
        error[:, kk] = error_tmp
    
    end
    pos = argmin(mean(error, dims=2))

    return C[pos]

end;