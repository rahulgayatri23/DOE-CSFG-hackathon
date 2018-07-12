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
#include "GPPComplex.h"

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



void flagOCC_solver(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht, int ncouls, int igp)
{
    GPPComplex matngmatmgp = GPPComplex(0.0, 0.0);
    GPPComplex matngpmatmg = GPPComplex(0.0, 0.0);
    GPPComplex ssxa(0.00, 0.00);
    GPPComplex scha(0.00, 0.00);
    for(int ig=0; ig<ncouls; ++ig)
    {
        GPPComplex wtilde = wtilde_array[my_igp*ncouls+ig];
        GPPComplex wtilde2 = GPPComplex(wtilde);
        GPPComplex Omega2 = GPPComplex_product(wtilde2, I_eps_array[my_igp*ncouls+ig]);
        GPPComplex mygpvar1 = GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
        GPPComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        GPPComplex matngmatmgp = GPPComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
        if(ig != igp) matngpmatmg = GPPComplex_product(GPPComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);

        ssxt_scht_solver(wxt, igp, my_igp, ig, wtilde, wtilde2, Omega2, matngmatmgp, matngpmatmg, mygpvar1, mygpvar2, ssxa, scha, I_eps_array[my_igp*ncouls+ig]); 
        ssxt += ssxa;
        scht += scha;
    }
}

void noflagOCC_solver(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht, int ncouls, int igp)
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

    double to1 = 1e-6;
    double gamma = 0.5;
    double sexcut = 4.0;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);

    double e_n1kq= 6.0; 

    //Printing out the params passed.
    std::cout << "******************Running pure Cuda version of the code with : *************************" << std::endl;
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
    double achtemp_re[3], achtemp_im[3];
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

    //Data structures on Device
    GPPComplex *d_acht_n1_loc, *d_aqsmtemp, *d_aqsntemp, *d_I_eps_array, *d_wtilde_array, *d_asxtemp;
    double *d_achtemp_re, *d_achtemp_im, *d_vcoul, *d_wx_array, *d_achstemp_re, *d_achstemp_im, \
        achstemp_re, achstemp_im;
    int *d_inv_igp_index;

    CudaSafeCall(cudaMalloc((void**) &d_acht_n1_loc, number_bands*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsntemp, number_bands*ncouls*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_I_eps_array, ngpown*ncouls*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_wtilde_array, ngpown*ncouls*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_wx_array, 3*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_inv_igp_index, ngpown*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_vcoul, ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_re, 3*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_im, 3*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_asxtemp, 3*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_achstemp_re, sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achstemp_im, sizeof(double)));


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
       achtemp_re[iw] = 0.00; achtemp_im[iw] = 0.00;

        wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
        if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;
    }

    //Update data structures on device
    CudaSafeCall(cudaMemcpy(d_acht_n1_loc, acht_n1_loc, number_bands*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_I_eps_array, I_eps_array, ngpown*ncouls*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_wtilde_array, wtilde_array, ngpown*ncouls*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_inv_igp_index, inv_igp_index, ngpown*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_wx_array, wx_array, 3*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achtemp_re, achtemp_re, 3*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achtemp_im, achtemp_im, 3*sizeof(double), cudaMemcpyHostToDevice));

    auto startTimer = std::chrono::high_resolution_clock::now();

//    till_nvbandKernel(d_asxtemp, d_inv_igp_index, d_vcoul, d_wtilde_array, d_aqsmtemp, d_aqsntemp, d_I_eps_array, d_wx_array, nvband, ncouls, ngpown);
//
    d_reduce_achstemp(number_bands, d_inv_igp_index, ncouls, d_aqsmtemp, d_aqsntemp, d_I_eps_array, d_achstemp_re, d_achstemp_im, ngpown, d_vcoul);

    d_achtemp_kernel(number_bands, ngpown, ncouls, d_wtilde_array, d_inv_igp_index, d_aqsmtemp, d_aqsntemp, d_I_eps_array, d_acht_n1_loc, d_wx_array, d_vcoul, d_achtemp_re, d_achtemp_im);

    cudaDeviceSynchronize();
    CudaSafeCall(cudaMemcpy(&achstemp_re, d_achstemp_re, sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(&achstemp_im, d_achstemp_im, sizeof(double), cudaMemcpyDeviceToHost));

    CudaSafeCall(cudaMemcpy(achtemp_im, d_achtemp_im, 3*sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(achtemp_re, d_achtemp_re, 3*sizeof(double), cudaMemcpyDeviceToHost));

    std::chrono::duration<double> elapsedTimer = std::chrono::high_resolution_clock::now() - startTimer;

    GPPComplex achstemp(achstemp_re, achstemp_im);

    for(int iw=nstart; iw<nend; ++iw)
    {
        GPPComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
        achtemp[iw].print();
    }

    cout << "********** Time Taken **********= " << elapsedTimer.count() << " secs" << endl;

    free(acht_n1_loc);
    free(wtilde_array);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(inv_igp_index);
    free(vcoul);


    //Free Cuda memory
    cudaFree(d_acht_n1_loc);
    cudaFree(d_wtilde_array);
    cudaFree(d_aqsmtemp);
    cudaFree(d_aqsntemp);
    cudaFree(d_I_eps_array);
    cudaFree(d_inv_igp_index);
    cudaFree(d_vcoul);
    cudaFree(d_asxtemp);
    cudaFree(d_achstemp_re);
    cudaFree(d_achstemp_im);
    cudaFree(d_wx_array);
    cudaFree(d_achtemp_re);
    cudaFree(d_achtemp_im);

    return 0;
}

//Almost done code
