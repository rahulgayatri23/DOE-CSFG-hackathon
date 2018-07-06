/*
OpenMP3.5 version of the GPP code. All the double dimension arrays are represented as single dim and indexed accordingly for performance.
noflagOCC_solver is the main loop in which most of the computation happens.
The achtemp array, which holds the final output is created for each thread and the results are finally accumulated into one single array. This is done in order to avoid the "critical" which might otherwise be needed to maintain correctness. 
*/
#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <chrono>
#include "Complex.h"

using namespace std;


void ssxt_scht_solver(double wxt, int igp, int my_igp, int ig, GPPComplex wtilde, GPPComplex wtilde2, GPPComplex Omega2, GPPComplex matngmatmgp, GPPComplex matngpmatmg, GPPComplex mygpvar1, GPPComplex mygpvar2, GPPComplex& ssxa, GPPComplex& scha, GPPComplex I_eps_array_igp_myIgp)
{
    GPPComplex expr0( 0.0 , 0.0);
    double delw2, scha_mult, ssxcutoff;
    double to1 = 1e-6;
    double sexcut = 4.0;
    double gamma = 0.5;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    GPPComplex sch, ssx;

    GPPComplex wdiff = doubleMinusGPPComplex(wxt , wtilde);

    GPPComplex cden = wdiff;
    double rden = 1/GPPComplex_real(GPPComplex_product(cden , GPPComplex_conj(cden)));
    GPPComplex delw = GPPComplex_mult(GPPComplex_product(wtilde , GPPComplex_conj(cden)) , rden);
    double delwr = GPPComplex_real(GPPComplex_product(delw , GPPComplex_conj(delw)));
    double wdiffr = GPPComplex_real(GPPComplex_product(wdiff , GPPComplex_conj(wdiff)));

    if((wdiffr > limittwo) && (delwr < limitone))
    {
        sch = GPPComplex_product(delw , I_eps_array_igp_myIgp);
        double cden = std::pow(wxt,2);
        rden = std::pow(cden,2);
        rden = 1.00 / rden;
        ssx = GPPComplex_mult(Omega2 , cden , rden);
    }
    else if (delwr > to1)
    {
        sch = expr0;
        cden = GPPComplex_mult(GPPComplex_product(wtilde2, doublePlusGPPComplex((double)0.50, delw)), 4.00);
        rden = GPPComplex_real(GPPComplex_product(cden , GPPComplex_conj(cden)));
        rden = 1.00/rden;
        ssx = GPPComplex_product(GPPComplex_product(-Omega2 , GPPComplex_conj(cden)), GPPComplex_mult(delw, rden));
    }
    else
    {
        sch = expr0;
        ssx = expr0;
    }

    ssxcutoff = sexcut*GPPComplex_abs(I_eps_array_igp_myIgp);
    if((GPPComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;
    ssxa = GPPComplex_product(matngmatmgp,ssx);
    scha = GPPComplex_product(matngmatmgp,sch);
}

void reduce_achstemp(int n1, int* inv_igp_index, int ncouls, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex& achstemp, int ngpown, double* vcoul)
{
    double to1 = 1e-6;
    GPPComplex schstemp(0.0, 0.0);;
    for(int my_igp = 0; my_igp< ngpown; my_igp++)
    {
        GPPComplex schs(0.0, 0.0);
        GPPComplex matngmatmgp(0.0, 0.0);
        GPPComplex matngpmatmg(0.0, 0.0);
        GPPComplex halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
        int igp = inv_igp_index[my_igp];
        if(igp >= ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0)){

        GPPComplex mygpvar1 = GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
        GPPComplex mygpvar2 = aqsntemp[n1*ncouls+igp];

            schs = -I_eps_array[my_igp*ncouls+igp];
            matngmatmgp = GPPComplex_product(aqsntemp[n1*ncouls+igp] , mygpvar1);

            if(GPPComplex_abs(schs) > to1)
                GPPComplex_fma(schstemp, matngmatmgp, schs);
        }
        else
        {
            for(int ig=1; ig<ncouls; ++ig)
            {
                GPPComplex mult_result(GPPComplex_product(I_eps_array[my_igp*ncouls+ig] , mygpvar1));
                GPPComplex_fms(schstemp,aqsntemp[n1*ncouls+igp], mult_result); 
            }
        }
        achstemp += GPPComplex_mult(schstemp , vcoul[igp] * 0.5);
    }
}



void flagOCC_solver(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht, int ncouls, int igp, GPPComplex *ssxa, GPPComplex* scha)
{
    GPPComplex matngmatmgp = GPPComplex(0.0, 0.0);
    GPPComplex matngpmatmg = GPPComplex(0.0, 0.0);
    for(int ig=0; ig<ncouls; ++ig)
    {
        GPPComplex wtilde = wtilde_array[my_igp*ncouls+ig];
        GPPComplex wtilde2 = GPPComplex(wtilde);
        GPPComplex Omega2 = GPPComplex_product(wtilde2, I_eps_array[my_igp*ncouls+ig]);
        GPPComplex mygpvar1 = GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
        GPPComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        GPPComplex matngmatmgp = GPPComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
        if(ig != igp) matngpmatmg = GPPComplex_product(GPPComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);

        ssxt_scht_solver(wxt, igp, my_igp, ig, wtilde, wtilde2, Omega2, matngmatmgp, matngpmatmg, mygpvar1, mygpvar2, ssxa[ig], scha[ig], I_eps_array[my_igp*ncouls+ig]); 
        ssxt += ssxa[ig];
        scht += scha[ig];
    }
}

void noflagOCC_solver(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht, int ncouls, int igp, GPPComplex *scha)
{
    double to1 = 1e-6;
    double sexcut = 4.0;
    double gamma = 0.5;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    GPPComplex mygpvar1 = GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
    GPPComplex scht_loc(0.00, 0.00);
    
    for(int ig = 0; ig<ncouls; ++ig)
    {
        GPPComplex wdiff = doubleMinusGPPComplex(wxt, wtilde_array[my_igp*ncouls+ig]);
        double wdiffr = GPPComplex_real(GPPComplex_product(wdiff , GPPComplex_conj(wdiff)));
        double rden = 1/wdiffr;

        GPPComplex delw = GPPComplex_mult(GPPComplex_product(wtilde_array[my_igp*ncouls+ig] , GPPComplex_conj(wdiff)) ,rden); 
        double delwr = GPPComplex_real(GPPComplex_product(delw , GPPComplex_conj(delw)));

        scht_loc += GPPComplex_product(GPPComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]) , GPPComplex_product(delw , I_eps_array[my_igp*ncouls+ig])) ;
    }

    scht = scht_loc;
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <matrix_divider> " << endl;
        exit (0);
    }
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);

    int npes = 1; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    double e_lk = 10;
    double dw = 1;
    int nstart = 0, nend = 3;

    int tid, numThreads;
    #pragma omp parallel shared(numThreads) private(tid)
    {
            tid = omp_get_thread_num();
            if(tid == 0)
                numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

    double to1 = 1e-6;
    double gamma = 0.5;
    double sexcut = 4.0;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);

    double e_n1kq= 6.0; 

    //Printing out the params passed.
        std::cout << "******************Running pure OpenMP version of the code with : *************************" << std::endl;
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart \
        << "\t gamma = " << gamma \
        << "\t sexcut = " << sexcut \
        << "\t limitone = " << limitone \
        << "\t limittwo = " << limittwo << endl;

    GPPComplex expr0( 0.0 , 0.0);
    GPPComplex expr( 0.5 , 0.5);
    GPPComplex achtemp[3];
    GPPComplex *achtemp_threadArr = new GPPComplex [numThreads * 3];
    GPPComplex asxtemp[3];
    double wx_array[3];

    // Memory allocation
    GPPComplex *acht_n1_loc = new GPPComplex [number_bands];
    GPPComplex *aqsmtemp = new GPPComplex [number_bands*ncouls];
    GPPComplex *aqsntemp = new GPPComplex [number_bands*ncouls];
    GPPComplex *I_eps_array = new GPPComplex [ngpown*ncouls];
    GPPComplex *wtilde_array = new GPPComplex [ngpown*ncouls];

    int *inv_igp_index = new int[ngpown];
    double *vcoul = new double[ncouls];


    GPPComplex achstemp = GPPComplex(0.0, 0.0);
    GPPComplex ssx_array[3], \
        sch_array[3], \
        scht, ssxt, wtilde;

    GPPComplex *ssxa = new GPPComplex [ncouls];
    GPPComplex *scha = new GPPComplex [ncouls];

    double wxt;
    double occ=1.0;
    bool flag_occ;

    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsntemp[i*ncouls+j] = GPPComplex_mult(expr, (i+j));
           aqsmtemp[i*ncouls+j] = GPPComplex_mult(expr, (i+j));
       }


   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = GPPComplex_mult(expr, (i+j));
           wtilde_array[i*ncouls+j] = GPPComplex_mult(expr, (i+j));
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0*i;


    for(int ig=01; ig < ngpown; ++ig)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    for(int iw=nstart; iw<nend; ++iw)
    {
       achtemp[iw] = expr0;
        wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
        if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;
    }

    for(int i = 0; i < numThreads*3; ++i)
        achtemp_threadArr[i] = expr0;

    auto startTimer = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(3)
       for(int n1 = 0; n1 < nvband; n1++)
       {
            for(int my_igp=0; my_igp<ngpown; ++my_igp)
            {
               for(int iw=nstart; iw<nend; iw++)
               {
                    ssx_array[iw] = expr0;
                    int igp = inv_igp_index[my_igp];
                    if(igp >= ncouls)
                        igp = ncouls-1;

                    scht = ssxt = expr0;
                    wxt = wx_array[iw];
                    flagOCC_solver(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp, ssxa, scha);

                    ssx_array[iw] += ssxt;
                    sch_array[iw] += GPPComplex_mult(scht, 0.5);
                    asxtemp[iw] += GPPComplex_mult(GPPComplex_mult(ssx_array[iw] , occ) , vcoul[igp]);
               }
            }
       }




    for(int n1 = 0; n1<number_bands; ++n1)
    {
        reduce_achstemp(n1, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, ngpown, vcoul);

#pragma omp parallel for default(shared) private(scht, sch_array, ssx_array, tid) firstprivate(ngpown, ncouls) schedule(dynamic) 
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int igp = inv_igp_index[my_igp];
            if(igp >= ncouls)
                igp = ncouls-1;


            tid = omp_get_thread_num();

            for(int i=0; i<3; i++)
            {
                ssx_array[i] = expr0;
                sch_array[i] = expr0;
            }

            for(int iw=nstart; iw<nend; ++iw)
            {
                    scht = ssxt = expr0;
                    wxt = wx_array[iw];

                   noflagOCC_solver(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp, scha);

                    sch_array[iw] += GPPComplex_mult(scht, 0.5);
            }

            for(int iw=nstart; iw<nend; ++iw)
                achtemp_threadArr[tid*3 + iw] += GPPComplex_mult(sch_array[iw] , vcoul[igp]);

            acht_n1_loc[n1] += GPPComplex_mult(sch_array[2] , vcoul[igp]);

        }
    }
    std::chrono::duration<double> elapsedTimer = std::chrono::high_resolution_clock::now() - startTimer;

    for(int i = 0; i < numThreads; ++i)
    {
        for(int iw = nstart; iw < nend; ++iw)
            achtemp[iw] += achtemp_threadArr[i*3 + iw];
    }

    for(int iw=nstart; iw<nend; ++iw)
        achtemp[iw].print();

    cout << "********** Time Taken **********= " << elapsedTimer.count() << " secs" << endl;

    free(acht_n1_loc);
    free(wtilde_array);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(inv_igp_index);
    free(vcoul);
    free(achtemp_threadArr);

    return 0;
}

//Almost done code
