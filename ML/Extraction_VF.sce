function [dates,lnp,lnf,dln,f] = difflog(monnaie,i) // returns a matrix with the series of exchange rates and classical fundamentals  
    // monnaie : currency 
    // i : forecast horizon in months (i=1) 
    // dates : vector of dates (YYYYMM) 
    // lnp : series of logs of exchange rates at present time 
    // lnf : series of logs of exchange rates at future time (= present time + i months) 
    // dln : lnf - lnp 
    // f : matrix of classical fundamentals 
    
    N = 23; // number of experts
    donnees = read(monnaie + ".txt",-1,N + 2);
    T = length(donnees(:,1));

    annees = modulo(donnees($:-1:i+1,1),10000);
    mois = floor(modulo(donnees($:-1:i+1,1),1000000)/10000);
    dates = annees*100 + mois;

    donnees(:,2:$) = 100*log(donnees(:,2:$)); //log-transformation
    y = donnees($:-1:1,2); // actual rates
    experts = donnees($:-1:1,3:$); // fundamentals

    lnp = y(1:T-i); // present values
    lnf = y(i+1:T); // future values
    dln = lnf- lnp;
    f0 = experts - y*ones(1,N); // predictors of the exchange rate variation
    f = f0(1:$-i,1:N); // final set of predictors

    n = (N-1)/2;
    f = [f(:,1:n+1),f(:,1),f(:,n+2:N)]; // duplication of the random walk fundamental
endfunction

function [f] = extraireOutput(monnaie,i) // returns output gaps fundamentals
    // monnaie : currency
    // i : forecast horizon (i=1 month)
    // f : matrix of output gaps fundamentals
    f = read(strcat([monnaie,"out.txt"]),-1,8);
    f = f($:-1:i+1,:);
endfunction

function [f] = extraireTaylor(monnaie,i) // returns Taylor rule fundamentals
    // monnaie : currency
    // i : forecast horizon (i=1 month)
    // f : matrix of Taylor rule fundamentals   
    [dates,lnp,lnf,dln,experts] = difflog(monnaie,i);
    donnees = 100*log(read(strcat([monnaie,"tay.txt"]),-1,6));
    f = donnees($:-1:i+1,:) - lnp*ones(1,6);
endfunction

function [f] = extraireAbs(monnaie,i) // returns "absolute" fundamentals
    // monnaie : currency
    // i : forecast horizon (i=1)
    // f : matrix of absolute fundamentals
    [dates,lnp,lnf,dln,f0] = difflog(monnaie,i);
    donnees = 100*log(read(strcat([monnaie,"abs.txt"]),-1,3));
    f = donnees($:-1:i+1,:) - lnp*ones(1,3);
endfunction

function [z] = l(y,f) // square loss function
    // y : actual series
    // f : series of fundamentals
    // z : series of square losses (between y and f)
    [T,N] = size(f);
    Y = y*ones(1,N);
    z = (Y - f).*(Y - f);
endfunction

function [L] = pertesCumulees(y,f,t0) // returns the series of cumulative square losses with a training period of t0 months
    // y : actual series
    // f : series of fundamentals
    // t0 : beginning of the training period (in months)
    // L : series of cumulative square losses (between y and f) with a training period of t0 months
    [T,N] = size(f);
    L = cumsum(l(y(t0:T,:),f(t0:T,:)),"r");
endfunction

function [eqm] = EQM(y,f,t0) // returns the root mean square error with a training period of t0 months
    // y : actual series
    // f : series of fundamentals
    // t0 : beginning of the training period (in months)
    // eqm : root mean square error (between y and f) with a training period of t0 months
    T = length(y);
    L = pertesCumulees(y,f,t0);
    eqm = sqrt(L(T- t0 + 1,:)/(T - t0 + 1));
endfunction

function [yhat] =  prediction(f,p) // returns forecasts given a fundamentals and a weights vectors
    // f : vector of fundamentals
    // p : vector of weights
    // yhat : predictor agreggating (i.e. summing) f and p
    yhat = sum(f.*p,"c");
endfunction

function [i] = first_index(vect) // for a given vector, returns the first index that is not NaN
    // vect : a given vector
    // i ; the first index of vect that is not NaN
    T = length(vect);
    count = cumsum(isnan(vect));
    ref = (1:T)';
    i = min(find(count~=ref));
endfunction

function [i] = last_index(vect) // for a given vector, returns the first index from the end that is not NaN
    // vect : a given vector
    // i ; the first index from the end of vect that is not NaN    
    T = length(vect);
    new_vect = vect(T:-1:1);
    i = T - first_index(new_vect) + 1;
endfunction

function [dataset] = dataname(monnaie,experts,TYPE) // return the name of the data file to exploit to build data according to the currency, the set of experts, ... 
    // monnaie : currency
    // experts : list of integers corresponding to the fundamentals used ; 1 : Random Walk ; 3 : PPP ; 6 : 12-month growth of money quantity ; 8 : UIRP ; 9 : differential of interest rates ; 11 : 12-month growth rate of industrial production index
    // TYPE : "EOM" (end-of-month exchange rate); "RT" (real-time data); "UIRP_2018" (UIRP extended until 2018); "FRED" (average exchange rate)
    // dataset : name of the data file to exploit
    monnaies = ["GBP","JPY","CHF","CAN","SEK","DNK","AUD","FRF","DEM","ITL","NLG","PTE"];
    j = find(monnaies == monnaie);

    EOM_names = ["GBP_ae2014","JPY_ae2014","CHF_ae2014","CAN_ae2014","SEK_ae2014","DNK_ae2014","AUD_ae2014","FRF_ae","DEM_ae","ITL_ae","NLG_ae","PTE_ae";
    "GBP_ae2014","JPY_ae2014","CHF_ae2014uirp","CAN_ae2014","SEK_ae2014uirpmon","DNK_ae2014","AUD_ae2014uirp","FRF_ae","DEM_ae","ITL_ae","NLG_ae","PTE_aeuirp";
    "GBP_ae2014","JPY_ae2014mon","CHF_ae2014mon","CAN_ae2014mon","SEK_ae2014uirpmon","DNK_ae2014mon","AUD_ae2014mon","FRF_aemon","DEM_aemon","ITL_aemon","NLG_ae","PTE_aemon"];

    FRED_names = ["GBP_FRED","JPY_FRED","CHF_FRED","CAN_FRED","SEK_FRED","DNK_FRED","AUD_FRED","FRF_FRED","DEM_FRED","ITL_FRED","NLG_FRED","PTE_FRED";
    "GBP_FRED","JPY_FRED","CHF_FREDuirp","CAN_FRED","SEK_FREDuirpmon","DNK_FRED","AUD_FREDuirp","FRF_FRED","DEM_FRED","ITL_FRED","NLG_FRED","PTE_FREDuirp";
    "GBP_FRED","JPY_FREDmon","CHF_FREDmon","CAN_FREDmon","SEK_FREDuirpmon","DNK_FREDmon","AUD_FREDmon","FRF_FREDmon","DEM_FREDmon","ITL_FREDmon","NLG_FRED","PTE_FREDmon"];

    RT_names = ["GBP_rt2018","JPY_rt2018","CHF_rt2018","CAN_rt2018","SEK_rt2018","DNK_rt2018","AUD_rt2018";
    "GBP_rt2018","JPY_rt2018","CHF_rt2018","CAN_rt2018","SEK_rt2018","DNK_rt2018","AUD_rt2018";
    "GBP_rt2018","JPY_rt2018","CHF_rt2018","CAN_rt2018","SEK_rt2018","DNK_rt2018","AUD_rt2018"];

    UIRP2018_names = ["GBP_uirp2018","JPY_uirp2018","CHF_uirp2018","CAN_uirp2018","SEK_uirp2018","DNK_uirp2018","AUD_uirp2018"];

    if TYPE == "EOM" then      
        if (isequal(experts,[1,2]))|(isequal(experts,[1,3]))|(isequal(experts,[1,4])) then
            dataset = EOM_names(1,j);
        end
        if isequal(experts,[1,8]) then
            dataset = EOM_names(2,j);
        end
        if (isequal(experts,[1,6,11]))|(isequal(experts,[1,6,9,11]))|(isequal(experts,[1,3,6,8,9,11]))|(isequal(experts,[1,5,10]))|(isequal(experts,[1,5,10,8,9]))|(isequal(experts,[1,7,12]))|(isequal(experts,[1,7,12,8,9])) then
            dataset = EOM_names(3,j);
        end    
    end

    if TYPE == "FRED" then      
        if (isequal(experts,[1,3]))|(isequal(experts,[1,2]))|(isequal(experts,[1,4])) then
            dataset = FRED_names(1,j);
        end
        if isequal(experts,[1,8]) then
            dataset = FRED_names(2,j);
        end
        if (isequal(experts,[1,6,11]))|(isequal(experts,[1,6,9,11]))|(isequal(experts,[1,3,6,8,9,11]))|(isequal(experts,[1,5,10]))|(isequal(experts,[1,5,10,8,9]))|(isequal(experts,[1,7,12]))|(isequal(experts,[1,7,12,8,9])) then
            dataset = FRED_names(3,j);
        end    
    end

    if TYPE == "RT" then      
        if (isequal(experts,[1,3]))|(isequal(experts,[1,2]))|(isequal(experts,[1,4])) then
            dataset = RT_names(1,j);
        end
        if isequal(experts,[1,8]) then
            dataset = RT_names(2,j);
        end
        if (isequal(experts,[1,6,11]))|(isequal(experts,[1,6,9,11]))|(isequal(experts,[1,3,6,8,9,11]))|(isequal(experts,[1,5,10]))|(isequal(experts,[1,5,10,8,9]))|(isequal(experts,[1,7,12]))|(isequal(experts,[1,7,12,8,9])) then
            dataset = RT_names(3,j);
        end    
    end

    if TYPE == "UIRP2018" then
        dataset = UIRP2018_names(j);
    end
endfunction
