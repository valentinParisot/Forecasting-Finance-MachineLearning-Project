function [u] = ridgeEsc(i,y,f,lambda,bet,gam) // applies the discounted ridge strategy and returns the corresponding forecasting weights
    // i : forecast horizon in months
    // y : series to forecast
    // f : series of fundamentals
    // lambda : regularization factor of the ridge strategy
    // bet : discount "power" factor (bet = 2 generally)
    // gam : other discount factor
    // u : series of forecasting weights resulting from discounted ridge
    [T,N] = size(f);
    u = zeros(T,N);
    u(1:i,:) = 1/N*ones(i,N); // weights vector (initialized to uniform distribution)

    disc = sqrt(1+gam*(1:1:T).^(-bet));

    for t = (i+1):1:T

        b = disc(1:t-i)'.*y(t-i:-1:1,:);
        A = disc(1:t-i)'*ones(1,N).*f(t-i:-1:1,:);
        // constraints of the maximization problem (there are none here)
        C = []; 
        c = [];
        ci = [];
        cs = [];

        Q = lambda*eye(N,N) + A'*A;
        x = -A'*b;
        [v,lagr]=qld(Q,x,C,c,ci,cs,0); 

        u(t,:) = v';
    end

endfunction

function [u] = ridgeFen(i,y,f,lambda,H) // applies the rolling ridge strategy (when lambda = 0 this corresponds to rolling OLS)
    // i : forecast horizon in months
    // y : series to forecast
    // f : series of fundamentals
    // lambda : regularization factor of the ridge strategy (when lambda = 0 this corresponds to rolling OLS)
    // H : length of the window for rolling
    // u : series of forecasting weights resulting from rolling ridge
    [T,N] = size(f);
    u = 1/N*ones(i,N); // woeghts vector (initialized to uniform distribution)

    for t = (i+1):1:T
        b = y(max(t-i-H,1):t-i,:);
        A = f(max(t-i-H,1):t-i,:);
        // constraints of the maximization problem (there are none here)
        C = [];
        c = [];

        ci = [];
        cs = [];

        Q = lambda*eye(N,N) + A'*A;
        x = -A'*b;
        [v,lagr]=qld(Q,x,C,c,ci,cs,0); 

        u = [u; v']; 
    end

endfunction

function [p] = EWAEsc(i,y,f,eta,bet,gam) // applied the discounted exponentially weighted average strategy
    // i : forecast horizon in months
    // y : series to forecast
    // f : series of fundamentals
    // eta : learning rate of the EWA strategy
    // bet : discount "power" factor (bet = 2 generally)
    // gam : second discount factor
    // p : series of forecasting weights resulting from discounted EWA  
    [T,N] = size(f);
    w = [ones(i,N)];
    p = w./(sum(w,"c")*ones(1,N)); // weights vector (initialized to uniform distribution)

    if T>i then

        disc = 1+gam*(1:1:T-i).^(-bet);

        pertes = l(y(1:T-i),f(1:T-i,:));
        pertes_shift = pertes ;
        L = zeros(T-i,N);

        for t=1:1:(T-i)

            L = L + disc(t)*pertes_shift;
            pertes_shift = [zeros(1,N);pertes_shift(1:T-i-1,:)];
        end

        sqrt_etas = eta./(sqrt(1:T-i)'*ones(1,N));

        // "shifts" the cumulative losses vector if it is too big (so that exponentials can be computed)
        m = min(L,"c")*ones(1,N);
        L = L - (m>(200*sqrt_etas)).*(m-200*sqrt_etas);

        q = exp(- sqrt_etas.*L);
        q = q./(sum(q,"c")*ones(1,N));

        p = [p;q];

    end;

endfunction

function [p,lambdas,gammas] = adapt(i,y,f,strat,grille1,grille2,grille3) // applies the adaptive version of a given strategy (parameters vary over time and are picked at each time so that they maximize the past performance) 
    // i : forecast horizon (in months)
    // y : series to forecast
    // f : series of fundamentals
    // strat : strategy (EWAEsc or ridgeEsc)
    // grille1 : grid of values over which the first parameter of strat can vary (grille1 corresponds to the values of eta if strat = EWAEsc or to the values of lambda if strat = ridgeEsc)
    // grille2 : grid of values over which the second parameter of strat can vary (grille2 corresponds to the values of bet)
    // grille3 : grid of values over which the third parameter of strat can vary (grille3 corresponds to the values of gam)
    // p : series of forecasting weights resulting from the adaptive strategy
    // lambdas : values of grille1 that are picked by the adaptive strategy over time
    // gammas : values of grille3 that are picked by the adaptive strategy over time
    [T,N] = size(f);
    l1 = length(grille1);
    l2 = length(grille2);
    l3 = length(grille3);
    poids = zeros(T,N*l1*l2*l3);
    prev = zeros(T,l1*l2*l3);
    lambdas = ones(i,1)*grille1(1);
    gammas = ones(i,1)*grille3(1);

    // computes the weights resulting from strat for every combination of parameter values
    for j = 1:l1
        for k = 1:l2
            for m = 1:l3
                poids_j_k_m = strat(i,y,f,grille1(j),grille2(k),grille3(m))
                poids(:,N*((j-1)*l2*l3 + (k-1)*l3 + m-1)+1:N*((j-1)*l2*l3 + (k-1)*l3 + m)) = poids_j_k_m;
                prev(:,(j-1)*l2*l3 + (k-1)*l3 + m) = prediction(f,poids_j_k_m);
            end
        end
    end

    par1 = 1;
    par2 = 1;
    par3 = 1;

    p = strat(i,y(1:i),f(1:i,:),grille1(par1),grille2(par2),grille3(par3));
    pertes = l(y,prev);
    L = cumsum(pertes(1:T,:),"r");

    [mins,ind] = min(L(1:(T-i),:),"c"); // picks the values of the parameters that maximize the past performance of strat at eacht time t

    ind1 = int((ind-1)/(l2*l3))+1;
    r1 = pmodulo(ind - 1,l2*l3)+1;

    ind2 = int((r1-1)/l3)+1;
    ind3 = pmodulo(r1-1,l3) + 1;

    k = (ind1-1)*l2*l3 + (ind2-1)*l3 + ind3;

    ext_poids = poids((i+1):T,:)';
    vec_poids = (ext_poids(:))';

    k = k + (0:l1*l2*l3:(T-i-1)*l1*l2*l3)';
    pt = []; 
    for j = 1:N
        pt = [pt,vec_poids(N*(k-1)+j)'];
    end
    lambdas = grille1(ind1);
    gammas = grille3(ind3);
    p = [p;pt];

endfunction

// ********** Statistical tests **********

function [D_series] = DM_series(t0,dln,yhat_1,yhat_2) // returns the square loss differential series
    // t0 : beginning of the training period
    // dln : differential of the log of the exchange rates
    // yhat1 : first predictor for dln
    // yhat2 : second predictor for dln
    // D_series : square loss differential series
    D_series = (dln(t0:$)-yhat_1(t0:$)).^2 - (dln(t0:$)-yhat_2(t0:$)).^2;
endfunction   

function [A_series] = CW_series(t0,dln,yhat_1,yhat_2) // returns the Clark and West adjusted loss differential series
    // t0 : beginning of the training period
    // dln : differential of the log of the exchange rates
    // yhat1 : first predictor for dln
    // yhat2 : second predictor for dln
    // A_series : Clark and West adjusted loss differential series   
    A_series = DM_series(t0,dln,yhat_1,yhat_2) + (yhat_1(t0:$)-yhat_2(t0:$)).^2;
endfunction  

function [sDM,H,pDM]= calcDM(D) // computes the DM stat and the corresponding p-value given a loss differential series
    // D : square loss differential series
    // sDM : Diebold and Mariano statistics
    // H : value of the truncation lag used to compute the statistics
    // pDM : p-value of the DM statistics
    n = length(D);
    s = mean((D-mean(D)).^2);

    for h = 1:20 // 20 is about sqrt(n) 
        s = [s, s($)+2*sum((D(1:(n-h))-mean(D)).*(D((h+1):n)-mean(D)))/n]; 
    end
    [m,ind_H]=max(s); // chooses the truncation lag that maximizes s

    if sum(D.*D) == 0 then
        sDM = 0;
        H = -1 ;
    else
        sDM = sqrt(n)*mean(D)/sqrt(m);
        H = ind_H-1;
    end
    pDM = 1-cdfnor("PQ",sDM,0,1);
endfunction

function [sCW,pCW]=calcCW(A) // computes the CW stat and the corresponding p-value given an adjusted loss differential series
    // A : Clark and West adjusted loss differential series
    // sCW : Clark and West statistics
    // pCW : p-value of the CW statistics
    n=length(A);
    if sum(A.*A) == 0 then
        sCW = 0 ;
    else
        sCW = sqrt(n)*mean(A)/stdev(A);
    end
    pCW = 1-cdfnor("PQ",sCW,0,1); 
endfunction

function [pCW,pDM,sDM] = test_stat(t0,dln,yhat_1,yhat_2) // computes the DM and CW stats and p-values given two forecasting models
    // t0 : beginning of the training period
    // dln : differential of the log of the exchange rates
    // yhat1 : first predictor for dln
    // yhat2 : second predictor for dln
    // pCW : p-value of the Clark and West test
    // pDM : p-value of the Diebold and Mariano test
    // sDM : statistics resulting from the Diebold and Mariano test 
    [sDM,H,pDM] = calcDM(DM_series(t0,dln,yhat_1,yhat_2));
    [sCW,pCW] = calcCW(CW_series(t0,dln,yhat_1,yhat_2));
endfunction

function[percentage,sDM,pDM]=directional(t0,dln,yhat) // returns the percentage of directions of change correctly predicted and the DM stat and p-value associated
    // t0 : beginning of the training period
    // dln : differential of the log of the exchange rates
    // yhat : predictor for dln
    // percentage : percentage of directions of change correctly predicted by yhat
    // sDM : statistics resulting from the Diebold and Mariano test
    // pDM : p-value of the Diebold and Mariano test 
    D = 0. + ((dln(t0:$).*yhat(t0:$))>0);
    percentage = mean(D); // percentage of directions of change correctly predicted

    n = length(D);
    s = mean((D-mean(D)).^2);

    for h = 1:20 // 20 is about sqrt(n) 
        s = [s, s($)+2*sum((D(1:(n-h))-mean(D)).*(D((h+1):n)-mean(D)))/n]; 
    end
    [m,ind_H]=max(s); // chooses the truncation lag that maximizes s
    sDM = sqrt(n)*(percentage-0.5)/sqrt(m);
    H = ind_H-1;
    pDM = 1-cdfnor("PQ",sDM,0,1);
endfunction

function [vec_table2,vec_table4,vec_table6,vec_sDM] = build_tables(i,t0,dln,f,etas,lambdas,betas,gammas,w) // returns the tables of results
    // i : forecast horizon in months
    // t0 : beginning of the training period
    // dln : differential of the log of the exchange rates
    // f : fundamentals
    // etas : grid of learning rates for discounted EWA
    // lambdas : grid of regularization factors for discounted ridge
    // betas : values for the "power" discount factor
    // gammas : values for the second discount factor
    // w : length of the window used for rolling OLS
    // vec_table2 : matrix stocking the results (Theil ratio, DM p-stat, ...) of the different strategies (rolling OLS, recursive OLS, ridge, EWA)
    // vec_table4 : matrix stocking the results of comparisons between strategies (ridge vs rolling OLS, ridge vs recursive OLS, ...)
    // vec_table6 : matrix stocking the results for the directional tests
    // vec_sDM : matrix stocking the DM statistics computed
    [T,N] = size(f);

    rw_1 = zeros(T,1); // random walk
    eqm_rw = EQM(dln,rw_1,t0); // random walk RMSE

    // Rolling regression

    [p_rol] = ridgeFen(i,dln,f,0,w); 
    yhat_rol = prediction(f,p_rol); // forecasts
    eqm_rol = EQM(dln,yhat_rol,t0); // RMSE
    theil_rol = eqm_rol/eqm_rw ; // Theil ratio

    [pCW_rol,pDM_rol,sDM_rol] = test_stat(t0,dln,rw_1,yhat_rol); // CW p-value and DM stat and p-value       

    //Recursive regression

    [p_rec,lambdas_rec] = adapt(i,dln,f,ridgeEsc,[0],betas,[0]); 
    yhat_rec = prediction(f,p_rec); // forecasts
    eqm_rec = EQM(dln,yhat_rec,t0); // RMSE 
    theil_rec = eqm_rec/eqm_rw; // Theil ratio

    [pCW_rec,pDM_rec,sDM_rec] = test_stat(t0,dln,rw_1,yhat_rec); // CW p-value and DM stat and p-value

    // Ridge Esc (forecasts, RMSE, Theil ratio, CW, DM)

    [p_rid,lams,gams] = adapt(i,dln,f,ridgeEsc,lambdas,betas,gammas);
    yhat_rid = prediction(f,p_rid);
    eqm_rid = EQM(dln,yhat_rid,t0);
    theil_rid = eqm_rid/eqm_rw;

    [pCW_rid,pDM_rid,sDM_rid] = test_stat(t0,dln,rw_1,yhat_rid);

    // EWA Esc (forecasts, RMSE, Theil ratio, CW, DM)

    [p_ewa,ets,gams] = adapt(i,dln,f,EWAEsc,etas,betas,gammas);
    yhat_ewa = prediction(f,p_ewa);
    eqm_ewa = EQM(dln,yhat_ewa,t0);
    theil_ewa = eqm_ewa/eqm_rw;

    [pCW_ewa,pDM_ewa,sDM_ewa] = test_stat(t0,dln,rw_1,yhat_ewa);

    // RidgeEsc versus rolling regression
    theil_ridVrol = eqm_rid/eqm_rol; // Theil Ratio
    [pCW_ridVrol,pDM_ridVrol,sDM_ridVrol] = test_stat(t0,dln,yhat_rol,yhat_rid); // CW p-value and DM stat and p-value

    // RidgeEsc versus recursive regression
    theil_ridVrec = eqm_rid/eqm_rec; // Theil ratio
    [pCW_ridVrec,pDM_ridVrec,sDM_ridVrec] = test_stat(t0,dln,yhat_rec,yhat_rid); // CW p-value and DM stat and p-value

    // EWAEsc versus rolling regression
    theil_ewaVrol = eqm_ewa/eqm_rol;
    [pCW_ewaVrol,pDM_ewaVrol,sDM_ewaVrol] = test_stat(t0,dln,yhat_rol,yhat_ewa);

    // EWAEsc versus recursive regression
    theil_ewaVrec = eqm_ewa/eqm_rec;
    [pCW_ewaVrec,pDM_ewaVrec,sDM_ewaVrec] = test_stat(t0,dln,yhat_rec,yhat_ewa);

    //Directional tests (percentage correctly predicted, DM stat and p-value)

    [per_rol,sDM_dir_rol,pDM_dir_rol] = directional(t0,dln,yhat_rol); // rolling regression
    [per_rec,sDM_dir_rec,pDM_dir_rec] = directional(t0,dln,yhat_rec); // recursive regression  
    [per_rid,sDM_dir_rid,pDM_dir_rid] = directional(t0,dln,yhat_rid); // ridge Esc
    [per_ewa,sDM_dir_ewa,pDM_dir_ewa] = directional(t0,dln,yhat_ewa); // EWAEsc

    vec_table2 = [eqm_rw,theil_rol,pCW_rol,sDM_rol,pDM_rol,theil_rec,pCW_rec,sDM_rec,pDM_rec,theil_rid,pCW_rid,sDM_rid,pDM_rid,theil_ewa,pCW_ewa,sDM_ewa,pDM_ewa]; // results of the strategies (rolling, recursive, ridgeEsc, EWAEsc)
    vec_table4 = [theil_ridVrol,sDM_ridVrol,pDM_ridVrol,theil_ridVrec,sDM_ridVrec,pDM_ridVrec,theil_ewaVrol,sDM_ewaVrol,pDM_ewaVrol,theil_ewaVrec,sDM_ewaVrec,pDM_ewaVrec]; // results for the comparisons between strategies (RidgeEsc vs rolling, EWAEsc vs recursive, etc.)
    vec_table6 = [per_rol,sDM_dir_rol,pDM_dir_rol,per_rec,sDM_dir_rec,pDM_dir_rec,per_rid,sDM_dir_rid,pDM_dir_rid,per_ewa,sDM_dir_ewa,pDM_dir_ewa]; // directional tests results
    vec_sDM = [sDM_rol,sDM_rec,sDM_rid,sDM_ewa,sDM_ridVrol,sDM_ridVrec,sDM_ewaVrol,sDM_ewaVrec,sDM_dir_rol,sDM_dir_rec,sDM_dir_rid,sDM_dir_ewa]; // DM statistics (needed for bootstrapping)
endfunction

// ********** Economic criterion **********

function [w] = weights(rf_rate,foreign_rate,actual_variations,predicted_variations,sigma_p) // returns portfolio weights
    // rf_rate : series of risk-free rate
    // foreign_rate : matrix of foreign interest rates
    // actual_variations : matrix of actual variations of exchange rates
    // predicted_variations : given a forecasting strategy, matrix of predicted variations of exchange rates
    // sigma_p : target volatility
    // w : portfolio weights on the different currencies
    return_rate = foreign_rate - actual_variations;
    mu = foreign_rate - predicted_variations;
    [T,N] = size(mu);
    one = ones(1,N);
    w = zeros(T,N);

    for s= N+2:T
        Sigma = cov(return_rate(1:s-1,:)); // empirical covariance matrix
        C = (mu(s,:) - rf_rate(s,:)*one)*inv(Sigma)*(mu(s,:) - rf_rate(s,:)*one)'; 
        w(s,:) = sigma_p/sqrt(C)*(mu(s,:) - rf_rate(s,:)*one)*inv(Sigma);   
    end
endfunction

function [f,g,ind] = returns(lambda, ind, qq, sigma_p, c, D) // objective function for the optimization problem with constraint on the sum of square weights
    // lambda, ind, qq, sigma_p, c, D : global variables
    // f : objective function
    // g : gradient of the objective function 

    DL = lambda(1)*D+lambda(2)*eye(D);

    f = (1/2)*(qq'*inv(DL)*qq + lambda(1)*sigma_p^2 + lambda(2)*c);

    g = [0; 0];
    g(1) = 1/2*(- sum(qq.^2 ./ diag(DL).^2 .* diag(D)) + sigma_p^2) ;
    g(2) = 1/2*(- sum(qq.^2 ./ diag(DL).^2) + c) ;

endfunction  

function [f,g,ind] = returns_sum(lambda, ind, qq, sigma_p, c, D, P) // objective function for the optimization problem with constraint on the sum of the square weights and on the sum of the weights
    // lambda, ind, qq, sigma_p, c, D, P : global variables
    // f : objective function
    // g : gradient of the objective function 

    DL = lambda(1)*D+lambda(2)*eye(D);
    qqq = qq - lambda(3)*P'*ones(length(qq),1);

    f = (1/2)*(qqq'*inv(DL)*qqq + lambda(1)*sigma_p^2 + lambda(2)*c) + lambda(3);

    g = [0; 0; 0];
    g(1) = 1/2*(- sum(qqq.^2 ./ diag(DL).^2 .* diag(D)) + sigma_p^2) ;
    g(2) = 1/2*(- sum(qqq.^2 ./ diag(DL).^2) + c) ;
    g(3) = - sum(P*inv(DL)*qqq) + 1;

endfunction

function [w] = cons_weights(rf_rate,foreign_rate,actual_variations,predicted_variations,sigma_p,c,somme) // returns portfolio weights with user-defined constraints 
    // rf_rate : series of risk-free rate
    // foreign_rate : matrix of foreign interest rates
    // actual_variations : matrix of actual variations of exchange rates
    // predicted_variations : given a forecasting strategy, matrix of predicted variations of exchange rates
    // sigma_p : target volatility
    // c : maximal value of the sum of the square weights
    // somme : if %T (true) then the sum of the weights is constrained to be smaller than 1 ; if %F (false) there is no constraint on the sum of the weights
    // w : portfolio weights on the different currencies
    return_rate = foreign_rate - actual_variations;
    mu = foreign_rate - predicted_variations;
    [T,N] = size(mu);
    one = ones(1,N);
    w = zeros(T,N);

    for s= N+2:T
        Sigma = cov(return_rate(1:s-1,:)); // empirical covariance matrix
        [P,D] = spec(Sigma);
        q = (mu(s,:) - rf_rate(s,:)*one)' ;
        qq = P'*q;

        if ~ somme then // no constraint on the sum of the weights
            arg = list(returns, qq, sigma_p, c, D);
            [fopt,lopt] = optim(arg, "b", [10^-30;10^-30], [1000;1000], [0.1;0.1],"ar",1000,1000);
            w(s,:) = (inv(lopt(1)*Sigma + lopt(2)*eye(Sigma))*q)' ; ;
        else // the sum of the weights is constrained to be smaller than 1
            arg = list(returns_sum, qq, sigma_p, c, D,P);
            [fopt,lopt] = optim(arg, "b", [10^-30;10^-30;10^-30], [1000;1000;1000], [0.1;0.1;0.1],"ar",1000,1000);
            w(s,:) = (inv(lopt(1)*Sigma + lopt(2)*eye(Sigma))*(q-lopt(3)))' ; 
        end
    end
endfunction

function [Rp] = ptf_return(rf_rate,foreign_rate,actual_variations,predicted_variations,sigma_p,w) // returns the portfolio return obtained using portfolio weights w
    // rf_rate : series of risk-free rates
    // foreign_rate : matrix of foreign interest rates
    // actual_variations : matrix of actual variations of exchange rates
    // predicted_variations : given a forecasting strategy, matrix of predicted variations of exchange rates
    // sigma_p : target volatility
    // w : portfolio weights
    // Rp : portfolio returns
    [T,N] = size(w);
    one = ones(1,N);
    Rp = 1 + (1-w*one').*rf_rate + sum(w.*(foreign_rate - actual_variations),"c");
endfunction

function [U] = utility(R,delta,t0) // utility function used to compute performance fees
    // R : returns
    // delta : risk aversion of the utility function
    // t0 : beginning of the training period
    // U : value of utility
    vect = R - delta/(2*(1+delta))*(R.^2);
    U = sum(vect(t0:$));
endfunction

function [PF] = performance_fee(Rw,Rp,delta,t0); // returns the performance fee
    // Rw : portfolio returns obtained with random walk
    // Rp : portfolio returns obtained with the forecasting strategy
    // delta : ris aversion of the utility function
    // t0 : beginning of the training period
    // PF : performance fee
    T = length(Rw);
    c = utility(Rp,delta,t0)-utility(Rw,delta,t0);
    vectb = - 1 + delta/(1+delta)*Rp;
    b = sum(vectb(t0:$));
    vecta = - delta/(2*(1+delta))*ones(T,1);
    a = sum(vecta(t0:$));
    p = poly([c,b,a],"x","coeff");
    proots = roots(p);
    PF = proots(2); 
endfunction

function [PR] = premium_return(Rw,Rp,rf_rate,delta,t0); // returns the premium return
    // Rw : portfolio returns obtained with random walk
    // Rp : portfolio returns obtained with the forecasting strategy
    // rf_rate : risk-free rates
    // delta : ris aversion of the utility function
    // t0 : beginning of the training period
    // PR : premium return
    
    vect1 = (Rp./(1+rf_rate)).^(1-delta);
    v1 = vect1(t0:$);
    vect2 = (Rw./(1+rf_rate)).^(1-delta);
    v2 = vect2(t0:$);

    PR = 1/(1-delta)*(log(mean(v1))-log(mean(v2))); 
endfunction

function [SR] = sharpe_ratio(Rp,rf_rate,t0) // returns the Sharpe ratio
    // Rp : portfolio returns
    // rf_rate : risk-free rates
    // t0 : beginning of the training period
    // SR : Sharpe ratio
    num = Rp(t0:$)-1-rf_rate(t0:$);
    den = stdev(Rp(t0:$));
    SR = mean(num./den);
endfunction

function [SR] = sortino_ratio(Rp,rf_rate,t0) // returns the Sortino ratio 
    // Rp : portfolio returns
    // rf_rate : risk-free rates
    // t0 : beginning of the training period
    // SR : Sortino ratio
    num = Rp(t0:$)-1-rf_rate(t0:$);
    vec = Rp(t0:$);
    den = stdev(vec(vec<1));
    SR = mean(num./den);
endfunction
