clear;
clc;

//L = list([1,3],[1,8],[1,6,11],[1,3,6,8,9,11]); // list of models that are used for forecasting : [1,3] = PPP ; [1,8] = UIRP ; [1,6,11] = monetary model ; [1,3,6,8,9,11] = all fundamentals
L = list([1,3,6,8,9,11]); // list of models that are used for forecasting : [1,3] = PPP ; [1,8] = UIRP ; [1,6,11] = monetary model ; [1,3,6,8,9,11] = all fundamentals

//monnaies = ["GBP","JPY","CHF","CAN","SEK","DNK","AUD","FRF","DEM","ITL","NLG","PTE"]; // list of currencies ; you have to drop "FRF","DEM","ITL","NLG","PTE" for Real-Time data 
monnaies = ["CHF"]; // list of currencies ; you have to drop "FRF","DEM","ITL","NLG","PTE" for Real-Time data 

iter = 1000; // number of bootstrap iterations

path = "/Users/vale/Desktop/codes" // directory where the codes are stored ; data must be stored in a specific subdirectory of "path" entitled "Data" ; results must be stored in a "Res" subdirectory
tic()

chdir(path)
exec("Extraction_VF.sce");
exec("Strategies_VF.sce");
exec("Bootstrap_VF.sce");  

i = 1; // forecast horizon in months
t0 = 121; // length of the training period ; set t0 = 61 for Real-Time data 
w = 119; // length of the window for rolling OLS ; set w = 59 for Real-Time data

TYPE = "EOM"; // type of data : "EOM", "RT" (real time), UIRP2018 (UIRP on 1973-2017), "FRED" (average exchange rates)
COUPLED = %T; // set COUPLED = %T to use coupled fundamentals in the model or COUPLED = %F to use decoupled fundamentals

BOOTSTRAP = %F; // BOOTSTRAP = %T if p-values are computed by bootstraping and %F if they are computed using normal distribution c.d.f.
D = 10; // maximul lag of the exchange rate differentials used in the autoregressive equation
H = 30; // maximul lag of the fundamental differentials used in the autoregressive equation

begin = 0; // if begin = 0, forecast process begins at the earliest observation of the sample ; if you want to start forecast from month MM at year YYYY, set begin = YYYYMM

etas = [10^-4,2*10^-4,5*10^-4,10^-3,2*10^-3,5*10^-3,0.01,0.02,0.05,0.1,0.2,0.5,1]; // list of learning rates for discounted EWA
lambdas = [0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000]; // list of regularization factors for discounted ridge
betas = [2]; // "power" discount factor
gammas = [0,1,2,5,10,20,50,100,200,500,1000]; // list of second discount factors

table2 = []; // results of the strategies (rolling, recursive, ridgeEsc, EWAEsc)
table4 = []; // results for the comparisons between strategies (RidgeEsc vs rolling, EWAEsc vs recursive, etc.)
table6 = []; // directional tests results
//// quoi garder ?

for J = L
    disp(J);
    for monnaie = monnaies
        chdir(path + "\Data")
        disp(monnaie); disp(string(toc()/3600)+"h");
        dataset = dataname(monnaie,J,TYPE);
        [dates,lnp,lnf,dln,f1] = difflog(dataset,i);

        // sets the beginning of the data sample
        if begin ~=0  then
            ibeg = min(find(dates>=begin));
        else
            ibeg = 1 ;
        end

        f2 = [f1(ibeg:$,J),f1(ibeg:$,J+12)]; // basic fundamentals

        lnp = lnp(ibeg:$);
        lnf = lnf(ibeg:$);
        dln = dln(ibeg:$);

        if COUPLED == %T then
           f2 = f2(:,1:$/2) + f2(:,$/2+1:$); 
        end

        T = length(f2(:,1));

        code lines for calculations with Taylor fundamentals
        f3 = extraireTaylor(monnaie,i);
        K = [1,3];
        f3 = [f3(:,K),f3(:,K+3)]
        if COUPLED == %T then
        f3 = f3(:,1:$/2) + f3(:,$/2+1:$);
        end

        //T = length(f3(:,1));

        // code lines for calculations with output gaps fundamentals
        // f4 = extraireOutput(monnaie,i); 
        // f4 = [f4(:,ind),f4(:,ind+4)];
        // if COUPLED == %T then
        // f4 = f4(:,1:$/2) + f4(:,$/2+1:$);
        // end

        // code lines for calculations with absolute fundamentals
        // f5 = extraireAbs(monnaie,i); 
        // if COUPLED == %T then
        // f5 = f5(:,1:$/2) + f5(:,$/2+1:$);
        // end

        f = [f3]; // use f3, f4 or f5 to get the appropriate model (Taylor-rule, output gaps, absolute fundamentals)

        [vec_table2,vec_table4,vec_table6,vec_sDM] = build_tables(i,t0,dln,f,etas,lambdas,betas,gammas,w); // whole results for the different models and the different currencies

        if BOOTSTRAP == %T then

            sim_sDM = semi_par_bootstrap(iter,D,H,i,t0,dln,f,etas,lambdas,betas,gammas,w,COUPLED); // matrix of the DM statistics (the number of lines is the number of iterations)

            mat_sDM = ones(iter,1)*vec_sDM;
            boot_pval = 1/iter*sum((sim_sDM > mat_sDM),"r"); // the p-value is the proportion of the simulated DM statistics above the DM statistics computed with the observed data  

            vec_table2([5,9,13,17]) = boot_pval([1,2,3,4]); // p-values for the Theil ratios resulting from the direct application of the strategies
            vec_table4([3,6,9,12]) = boot_pval([5,6,7,8]); // p-values for the Theil ratios resulting from the comparisons between strategies
            vec_table6([3,6,9,12]) = boot_pval([9,10,11,12]); // p-values for directional tests

        end

        table2 = [table2;vec_table2];
        table4 = [table4;vec_table4];
        table6 = [table6;vec_table6];

        // writes the results in text files

        if COUPLED == %T then
            str_coup = "_coupled_";
        else
            str_coup = "_decoupled_";
        end

        if BOOTSTRAP == %T then
            str_boot = "_bootstrap";
        else
            str_boot = "";
        end

        title2 = "table2_" + TYPE + str_coup + str_boot + string(t0)+ ".txt" ;
        title4 = "table4_" + TYPE + str_coup + str_boot + string(t0)+ ".txt" ;
        title6 = "table6_" + TYPE + str_coup + str_boot + string(t0)+ ".txt" ;

        chdir(path + "\Res")
        fprintfMat(title2,table2);
        fprintfMat(title4,table4);
        fprintfMat(title6,table6);
    end

    // introduces lines of zeros to separate the results for the different sets of experts 
    N2 = length(table2(1,:));
    table2 = [table2;zeros(1,N2)];
    N4 = length(table4(1,:));
    table4 = [table4;zeros(1,N4)];
    N6 = length(table6(1,:));
    table6 = [table6;zeros(1,N6)];
end

disp(toc())

