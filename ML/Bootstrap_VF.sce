function [X,Y] = regressors(ds,z,j,d,h,c_trend) // for a given fundamental, returns the response variable and the regressors used for the pre-bootstrap regression
    // ds : differential of the logs of the exchange rates
    // z : fundamentals
    // j : index of the fundamental which will be bootstrapped
    // d : maximum lag used for ds in the regression
    // h : maximum lag used for the differential of z in the regression
    // c_trend : if c_trend = 1 then there is no constant and no trend included in the regressors ; if c_trend = 2 there is a constant but no trend ; if c_trend = 3 there is a constant and a trend
    // X : matrix of regressors used for the pre-bootstrap regression
    // Y : response variable for the pre-bootstrap regression
    
    [T,N] = size(z);
    dz = z(2:T,:)-z(1:T-1,:);
    vec = max(d+1,h+1):(T-1);
    Y = dz(vec,j);

    X = [z(vec,j)]; //lag of the fundamental

    for k = 1:1:d
        X = [X,ds(vec-k)]; // lags of the exchange rate differential
    end

    for c = 1:1:h
        X = [X,dz(vec-c,j)]; // lags of the fundamental differential
    end

    if c_trend == 2 then
        X = [X,ones(length(vec),1)];
    end

    if c_trend == 3 then
        X = [X,ones(length(vec),1),(vec+1)'];
    end
endfunction

function [a,resi,AIC] = OLS_cons(ds,z,j,d,h,c_trend) // returns the coefficients, the residuals and the AIC for the pre-bootstrap regression
    // ds : differential of the logs of the exchange rates
    // z : fundamentals
    // j : index of the fundamental which will be bootstrapped
    // d : maximum lag used for ds in the regression
    // h : maximum lag used for the differential of z in the regression
    // c_trend : if c_trend = 1 then there is no constant and no trend included in the regression ; if c_trend = 2 there is a constant but no trend ; if c_trend = 3 there is a constant and a trend
    // a : estimated coefficients of the linear regression
    // resi : vector of residuals of the regression
    // AIC : Akaike information criterion
    [X,Y] = regressors(ds,z,j,d,h,c_trend);
    [n,p] = size(X);

    a = lsq(X,Y); 
    resi = Y-X*a;
    SSR = resi'*resi;
    sigma_2 = SSR/n;
    AIC = 2*p + n*log(2*%pi) + n*log(sigma_2)+SSR/sigma_2; 
endfunction

function [ind_sim] = rand_ind(T,q,n) // returns a vector of length n of integers drawn randomly between 1 and T by blocks of length q
    // T : maximal range of the integers drawn randomly
    // q : length of the blocks by which the integers are drawn
    // n : length of the random vector
    // ind_sim : vector gathering the integers drawn randomly

    m=ceil(n/q);
    ind_sim = grand(1,m,"uin",1,T - (q-1)); 

    for k=1:q-1
        ind_sim = [ind_sim; ind_sim(k,:)+1];
    end
    ind_sim = ind_sim(:);
    ind_sim = ind_sim(1:n); // ensures the final vector is of length n
endfunction

function [sim_sDM] = semi_par_bootstrap(iter,D,H,i,t0,dln,f,etas,lambdas,betas,gammas,w,COUPLED) // applies bootstrap procedure (iter is the number of iterations) : runs rolling OLS, recursive OLS, RidgeEsc and EWAEsc at each iteration and returns the vector of DM statistics 

    // iter : number of iterations for bootstrap
    // D : maximum value of d, lag order of the exchange rate differential in the AR regression (D=10)
    // H : maximum value of h, lag order of the fundamental differential (H = 30)
    // i : forecast horizon (i = 1 month)
    // t0 : beginning of the training period 
    // dln : vector of differences of the log of the exchange rates (s_t - s_{t-1})
    // f : matrix of fundamentals, f_{j,t} is a forecast of s_{t+1} - s_t
    // etas : grid of values for the parameter \eta in EWAEsc adaptative strategy
    // lambdas : grid of values for the parameter \lambda in ridgeEsc adaptative strategy
    // betas : grid of values for the power discount factor
    // gammas : grid of values for the second discount factor
    // w : length of the window for rolling OLS
    // COUPLED : if COUPLED = %T we have coupled fundamentals else we have decoupled ones
    
    // sim_sDM: the vector of the DM statistics computed at each bootstrap iteration

    sim_sDM = []; // vector of DM statistics

    disc_obs = 100 ; // number of observations that will be discarded in the beginning of the sample to minimize the influence of the choice of the "seeds"
    
    [T,N] = size(f);
    
    T_new = T + disc_obs ;
    
    h_min = zeros(1,N);
    d_min = zeros(1,N);
    c_trend_min = zeros(1,N);

    if COUPLED == %T then
        ind_J = 2:N ;
    else
        ind_J = [(2:1:N/2),(N/2+2:1:N)];
    end

    for j = ind_J // for each fundamental, computes h,d and c_trend that minimize AIC

        Mat_AIC = hypermat([D,H,3],zeros(1,D*H*3));

        for d = 1:1:D
            for h = 1:1:H
                for c_trend = 1:1:3
                    [a,resi,AIC] = OLS_cons(dln,f,j,d,h,c_trend);
                    Mat_AIC(d,h,c_trend) = AIC ;
                end
            end
        end
        [m,ind] = min(Mat_AIC);
        d_min(j) = ind(1);
        h_min(j) = ind(2);
        c_trend_min(j) = ind(3);
    end

    start = max([d_min,h_min])+2; // time at which begins the "recursion" (you need to initialize the first lagged values)

    for p =1:1:iter

        q = 1; // length of the blocks drawn randomly

        f_sim = zeros(T_new,N); // simulated matrix of fundamentals
        df_sim = zeros(T_new-1,N); // simulated differences of fundamentals
        dln_sim = zeros(T_new,N); // simulated series of exchange rates

        ind_init = rand_ind(T,q,start-1); // indices of the first values (drawn randomly)
        ind_sim = rand_ind(T-start+1,q,T_new-start+1) ; // indices of the values drawn randomly after time "start"

        ind_sim_dln = ind_sim + start-1 ;
        dln_sim = dln([ind_init; ind_sim_dln]); // exchange rate series is simulated according to the indices drawn randomly

        f_sim(1:start-1,:) = f(ind_init,:); // initiation of the fundamental simulated series
        df_sim(1:start-2,:) = f_sim(2:start-1,:)-f_sim(1:start-2,:);

        for j = ind_J

            [a,resi,AIC] = OLS_cons(dln,f,j,d_min(j),h_min(j),c_trend_min(j)); // coefficients and residuals of the regression used to simulate recursively the fundamental

            start_j = max(d_min(j),h_min(j)) + 2; 
            resi = resi(start - start_j + 1:$); // the residuals series is shifted to begin at t = start

            res_dz = resi(ind_sim); // fundamental series is simulated according to the indices drawn randomly (in the same order as exchange rates)

            c_trend_vect = [];

            if c_trend_min(j) == 2 then
                c_trend_vect = ones(T_new-start+2,1);
            end

            if c_trend_min(j) == 3 then
                c_trend_vect = [ones(T_new-start+2,1),(start:T_new+1)'];
            end

            vect = [f_sim(start-1,j),(dln_sim(start-2:-1:start-d_min(j)-1))',(df_sim(start-2:-1:start-h_min(j)-1,j))',c_trend_vect(1,:)]; // vector of regressors at time t

            for t = start:1:T_new 
                df_sim(t-1,j) = vect*a + res_dz(t-start+1); // the differential of the fundamental is computed according to the autoregressive equation and the simulated residuals
                f_sim(t,j) = f_sim(t-1,j)+df_sim(t-1,j); // value of the fundamental
                vect = [f_sim(t,j),dln_sim(t-1),vect(2:d_min(j)),df_sim(t-1,j),vect(d_min(j)+2:d_min(j)+h_min(j)),c_trend_vect(t-start+2,:)]; // vector of regressors is updated for the next time
            end
        end

        // the disc_obs = 100 first observations are discarded
        dln_sim = dln_sim(disc_obs + 1:$);
        f_sim = f_sim(disc_obs + 1:$,:);

        [vec_table2,vec_table4,vec_table6,vec_sDM] = build_tables(i,t0,dln_sim,f_sim,etas,lambdas,betas,gammas,w); // tables of results

        sim_sDM = [sim_sDM; vec_sDM]; // table of DM statistics
    end
endfunction
