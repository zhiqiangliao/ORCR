using JuMP, MosekTools

# The function argument "func" includes "Cost": convex function with beta >= 0;
#                       "Prod": concave function with beta >= 0;
#                       "Convex": convex function.


function CNLS(X::Matrix, y::Vector; func::String = "Convex")
    # Function to compute the Convex Nonparametric Least Square (CNLS) model 
    #               given output y, inputs X.
    # Zhiqiang Liao, Aalto University School of Business, Finland
    # Nov 18, 2022

    n = size(X, 1)
    p = size(X, 2)

    # model= Model(Mosek.Optimizer)
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # define the variables
    @variable(model, alpha[1:n])
    if func == "Convex" || func == "Concave"
        @variable(model, beta[1:n,1:p])
    elseif func == "Prod" || func == "Cost"
        @variable(model, beta[1:n,1:p]>=0)
    else
        @assert false "Function argument is false"
    end
    @variable(model, epsilon[1:n])
    
    # regression function
    for i in 1:n
        @constraint(model, y[i] == alpha[i] + X[i,:]'*beta[i,:] + epsilon[i] )
    end

    # constriant: Afriat inequality
    for i in 1:n
        for j in 1:n
            if( j != i)
                if func == "Concave" || func == "Prod"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] <= alpha[j] + X[i,:]'*beta[j,:] )
                elseif func == "Convex" || func == "Cost"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] >= alpha[j] + X[i,:]'*beta[j,:] )
                end
            end
        end
    end

    # objective function
    @objective(model, Min, sum(epsilon[i]^2 for i=1:n) )
    optimize!(model)

    # return estimates
    return value.(alpha), value.(beta)

end;

function pCNLS(X::Matrix, y::Vector, P::Real; func::String="Convex")
    # Function to compute the Convex Nonparametric Least Square (CNLS) model 
    #               given output y, inputs X.
    # Zhiqiang Liao, Aalto University School of Business, Finland
    # Nov 18, 2022

    n = size(X, 1)
    p = size(X, 2)

    model= Model(Mosek.Optimizer)
    set_silent(model)

    # define the variables
    @variable(model, alpha[1:n])
    if func == "Convex" || func == "Concave"
        @variable(model, beta[1:n,1:p])
    elseif func == "Prod" || func == "Cost"
        @variable(model, beta[1:n,1:p]>=0)
    else
        @assert false "Function argument is false"
    end
    @variable(model, epsilon[1:n])
    
    # regression function
    for i in 1:n
        @constraint(model, y[i] == alpha[i] + X[i,:]'*beta[i,:] + epsilon[i] )
    end

    # constriant 1: L2 norm
    @constraint(model, sum(beta[i,:]'*beta[i,:] for i in 1:n) <= P )

    # constriant: Afriat inequality
    for i in 1:n
        for j in 1:n
            if( j != i)
                if func == "Concave" || func == "Prod"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] <= alpha[j] + X[i,:]'*beta[j,:] )
                elseif func == "Convex" || func == "Cost"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] >= alpha[j] + X[i,:]'*beta[j,:] )
                end
            end
        end
    end

    # objective function
    @objective(model, Min, sum(epsilon[i]^2 for i=1:n) )
    optimize!(model)

    # return estimates
    return value.(alpha), value.(beta)

end;

function LCR(X::Matrix, y::Vector, L::Real; func::String="Convex")
    # Function to compute the lipschitz convex regression (LCR) model 
    #               given output y, inputs X.
    # Zhiqiang Liao, Aalto University School of Business, Finland
    # Mar 16, 2023

    n = size(X, 1)
    p = size(X, 2)

    model= Model(Mosek.Optimizer)
    # model = direct_model(CPLEX.Optimizer())
    set_silent(model)

    # define the variables
    @variable(model, alpha[1:n])
    if func == "Convex" || func == "Concave"
        @variable(model, beta[1:n,1:p])
    elseif func == "Prod" || func == "Cost"
        @variable(model, beta[1:n,1:p]>=0)
    else
        @assert false "Function argument is false"
    end
    @variable(model, epsilon[1:n])
    
    # regression function
    for i in 1:n
        @constraint(model, y[i] == alpha[i] + X[i,:]'*beta[i,:] + epsilon[i] )
    end
    
    # constriant 1: lipschitz bound
    for i in 1:n
        @constraint(model, sum(beta[i,j]^2 for j in 1:p) <= L^2 )
    end

    # constriant: Afriat inequality
    for i in 1:n
        for j in 1:n
            if( j != i)
                if func == "Concave" || func == "Prod"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] <= alpha[j] + X[i,:]'*beta[j,:] )
                elseif func == "Convex" || func == "Cost"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] >= alpha[j] + X[i,:]'*beta[j,:] )
                end
            end
        end
    end

    # objective function
    @objective(model, Min, sum(epsilon[i]^2 for i=1:n) )
    optimize!(model)

    # return estimates
    return value.(alpha), value.(beta)

end;

function ALCR(X::Matrix, y::Vector, L::Real; beta_0=nothing, func::String="Convex")
    # Function to compute the lipschitz convex regression (LCR) model 
    #               given output y, inputs X, beta_0.
    # Zhiqiang Liao, Aalto University School of Business, Finland
    # Mar 16, 2023

    n = size(X, 1)
    p = size(X, 2)

    model= Model(Mosek.Optimizer)
    set_silent(model)

    # define the variables
    @variable(model, alpha[1:n])
    if func == "Convex" || func == "Concave"
        @variable(model, beta[1:n,1:p])
    elseif func == "Prod" || func == "Cost"
        @variable(model, beta[1:n,1:p]>=0)
    else
        @assert false "Function argument is false"
    end
    @variable(model, epsilon[1:n])
    
    # regression function
    for i in 1:n
        @constraint(model, y[i] == alpha[i] + X[i,:]'*beta[i,:] + epsilon[i] )
    end
    
    # constriant 1: lipschitz bound
    for i in 1:n
        @constraint(model, sum((beta[i,j]-beta_0[j])^2 for j in 1:p) <= L^2 )
    end

    # constriant: Afriat inequality
    for i in 1:n
        for j in 1:n
            if( j != i)
                if func == "Concave" || func == "Prod"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] <= alpha[j] + X[i,:]'*beta[j,:] )
                elseif func == "Convex" || func == "Cost"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] >= alpha[j] + X[i,:]'*beta[j,:] )
                end
            end
        end
    end

    # objective function
    @objective(model, Min, sum(epsilon[i]^2 for i=1:n) )
    optimize!(model)

    # return estimates
    return value.(alpha), value.(beta)

end;

function WRCR(X::Matrix, y::Vector, w::Float64; beta_1=nothing, func::String="Convex")
    # Function to compute the weight restricted Convex Nonparametric Least Square (wrCNLS) model 
    #               given output y, inputs X.
    # Zhiqiang Liao, Aalto University School of Business, Finland
    # Mar 16, 2023

    n = size(X, 1)
    p = size(X, 2)

    model= Model(Mosek.Optimizer)
    set_silent(model)

    # define the variables
    @variable(model, alpha[1:n])
    if func == "Convex" || func == "Concave"
        @variable(model, beta[1:n,1:p])
    elseif func == "Prod" || func == "Cost"
        @variable(model, beta[1:n,1:p]>=0)
    else
        @assert false "Function argument is false"
    end
    @variable(model, epsilon[1:n])
    
    # regression function
    for i in 1:n
        @constraint(model, y[i] == alpha[i] + X[i,:]'*beta[i,:] + epsilon[i] )
    end
    
    # constriant 1: upper bound
    for i in 1:n
        for j in 1:p
            @constraint(model, beta[i,j] <= quantile(beta_1[:,j], 1-w))
        end
    end
    
    # constriant 2: lower bound
    for i in 1:n
        for j in 1:p
            @constraint(model, beta[i,j] >= quantile(beta_1[:,j], w))
        end
    end

    # constriant: Afriat inequality
    for i in 1:n
        for j in 1:n
            if( j != i)
                if func == "Concave" || func == "Prod"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] <= alpha[j] + X[i,:]'*beta[j,:] )
                elseif func == "Convex" || func == "Cost"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] >= alpha[j] + X[i,:]'*beta[j,:] )
                end
            end
        end
    end

    # objective function
    @objective(model, Min, sum(epsilon[i]^2 for i=1:n) )
    optimize!(model)

    # return estimates
    return value.(alpha), value.(beta)

end;

function AWRCR(X::Matrix, y::Vector, w::Float64; beta_0=nothing, beta_1=nothing, func::String="Convex")
    # Function to compute the weight restricted Convex Nonparametric Least Square (wrCNLS) model 
    #               given output y, inputs X.
    # Zhiqiang Liao, Aalto University School of Business, Finland
    # Mar 16, 2023

    n = size(X, 1)
    p = size(X, 2)

    model= Model(Mosek.Optimizer)
    set_silent(model)

    # define the variables
    @variable(model, alpha[1:n])
    if func == "Convex" || func == "Concave"
        @variable(model, beta[1:n,1:p])
    elseif func == "Prod" || func == "Cost"
        @variable(model, beta[1:n,1:p]>=0)
    else
        @assert false "Function argument is false"
    end
    @variable(model, epsilon[1:n])
    
    # regression function
    for i in 1:n
        @constraint(model, y[i] == alpha[i] + X[i,:]'*beta[i,:] + epsilon[i] )
    end
    
    # constriant 1: upper bound
    for i in 1:n
        for j in 1:p
            @constraint(model, beta[i,j] <= quantile(beta_1[:,j], w) + beta_0[j])
        end
    end
    
    # constriant 2: lower bound
    for i in 1:n
        for j in 1:p
            @constraint(model, beta[i,j] >= beta_0[j] - quantile(beta_1[:,j], w)) 
        end
    end

    # constriant: Afriat inequality
    for i in 1:n
        for j in 1:n
            if( j != i)
                if func == "Concave" || func == "Prod"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] <= alpha[j] + X[i,:]'*beta[j,:] )
                elseif func == "Convex" || func == "Cost"
                    @constraint(model, alpha[i] + X[i,:]'*beta[i,:] >= alpha[j] + X[i,:]'*beta[j,:] )
                end
            end
        end
    end

    # objective function
    @objective(model, Min, sum(epsilon[i]^2 for i=1:n) )
    optimize!(model)

    # return estimates
    return value.(alpha), value.(beta)

end;

function LIN(X::Matrix, y::Vector)
    # Function for linear regression model 
    #               given output y, inputs X.
    # Zhiqiang Liao, Aalto University School of Business, Finland
    # Mar 16, 2023

    n = size(y, 1)
    p = size(X, 2)

    model= Model(Mosek.Optimizer)
    set_silent(model)

    # define the variables
    @variable(model, alpha)
    @variable(model, beta[1:p])

    # objective function
    @objective(model, Min, sum((y[i] - (X[i,:]' * beta + alpha))^2 for i=1:n) )
    optimize!(model)

    # return estimates
    return value.(alpha), value.(beta)

end