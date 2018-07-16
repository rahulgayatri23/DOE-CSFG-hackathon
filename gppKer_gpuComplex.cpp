#include <iostream>
#include <cstdlib>
#include <memory>

#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>

#include "Complex.h"
#define nstart 0
#define nend 3

using namespace std;
int debug = 0;

inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, GPPComplex  *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads)
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

            GPPComplex mygpvar2, mygpvar1;
            mygpvar1 = GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
            mygpvar2 = aqsntemp[n1*ncouls+igp];



            schs = I_eps_array[my_igp*ncouls+igp];
            matngmatmgp = GPPComplex_product(mygpvar1, aqsntemp[n1*ncouls+igp]);


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

        schstemp = GPPComplex_mult(schstemp, vcoul[igp], 0.5);
        achstemp += schstemp;
    }
}

inline void flagOCC_solver(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht,int ncouls, int igp, int number_bands, int ngpown)
{
    GPPComplex expr0(0.00, 0.00);
    GPPComplex expr(0.5, 0.5);
    GPPComplex matngmatmgp(0.0, 0.0);
    GPPComplex matngpmatmg(0.0, 0.0);

    for(int ig=0; ig<ncouls; ++ig)
    {
        GPPComplex wtilde = wtilde_array[my_igp*ncouls+ig];
        GPPComplex wtilde2 = GPPComplex_square(wtilde);
        GPPComplex Omega2 = GPPComplex_product(wtilde2,I_eps_array[my_igp*ncouls+ig]);
        GPPComplex mygpvar1 = GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
        GPPComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        GPPComplex matngmatmgp = GPPComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
        if(ig != igp) matngpmatmg = GPPComplex_product(GPPComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);

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
            sch = GPPComplex_product(delw , I_eps_array[my_igp*ngpown+ig]);
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
    
        ssxcutoff = GPPComplex_abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
        if((GPPComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

        ssxt += GPPComplex_product(matngmatmgp , ssx);
        scht += GPPComplex_product(matngmatmgp , sch);
    }
}

void till_nvband(int number_bands, int nvband, int ngpown, int ncouls, int *inv_igp_index, double *wx_array, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex *asxtemp, double *vcoul)
{
    double occ = 1;
#pragma omp parallel for collapse(3)
       for(int n1 = 0; n1 < nvband; n1++)
       {
            for(int my_igp=0; my_igp<ngpown; ++my_igp)
            {
               for(int iw=nstart; iw<nend; iw++)
               {
                    int igp = inv_igp_index[my_igp];
                    if(igp >= ncouls)
                        igp = ncouls-1;

                    GPPComplex ssxt(0.00, 0.00);
                    GPPComplex scht(0.00, 0.00);
                    flagOCC_solver(wx_array[iw], wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp, number_bands, ngpown);
                    asxtemp[iw] += GPPComplex_mult(ssxt, occ , vcoul[igp]);
              }
            }
       }

}


void noFlagOCCSolver(int n1, int nvband, int ngpown, int ncouls, int *inv_igp_index, double *wx_array, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im)
{
#pragma omp parallel for  default(shared) 
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int igp = inv_igp_index[my_igp];
            if(igp >= ncouls)
                igp = ncouls-1;

            double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
            for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

            for(int ig = 0; ig<ncouls; ++ig)
            {
                for(int iw = nstart; iw < nend; ++iw)
                {
                    GPPComplex wdiff = doubleMinusGPPComplex(wx_array[iw], wtilde_array[my_igp*ncouls+ig]);
                    GPPComplex delw = GPPComplex_mult(GPPComplex_product(wtilde_array[my_igp*ncouls+ig] , GPPComplex_conj(wdiff)), 1/GPPComplex_real(GPPComplex_product(wdiff, GPPComplex_conj(wdiff)))); 
                    GPPComplex sch_array = GPPComplex_mult(GPPComplex_product(GPPComplex_product(GPPComplex_conj(aqsmtemp[n1*ncouls+igp]), aqsntemp[n1*ncouls+ig]), GPPComplex_product(delw , I_eps_array[my_igp*ncouls+ig])), 0.5*vcoul[igp]);
                    achtemp_re_loc[iw] += GPPComplex_real(sch_array);
                    achtemp_im_loc[iw] += GPPComplex_imag(sch_array);
                }
            }
            for(int iw = nstart; iw < nend; ++iw)
            {
#pragma omp atomic
                achtemp_re[iw] += achtemp_re_loc[iw];
#pragma omp atomic
                achtemp_im[iw] += achtemp_im_loc[iw];
            }
        } //ngpown
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
        exit (0);
    }
    auto start_totalTime = std::chrono::high_resolution_clock::now();

    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);


    int npes = 1; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task

    double e_lk = 10;
    double dw = 1;
    int inv_igp_index[ngpown];
    int indinv[ncouls+1];

    //OpenMP Printing of threads on Host and Device
    int tid, numThreads, numTeams;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;


    double to1 = 1e-6, \
    gamma = 0.5, \
    sexcut = 4.0;
    double limitone = 1.0/(to1*4.0), \
    limittwo = pow(0.5,2);

    double e_n1kq= 6.0; //This in the fortran code is derived through the double dimenrsion array ekq whose 2nd dimension is 1 and all the elements in the array have the same value

    //Printing out the params passed.
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


    //ALLOCATE statements from fortran gppkernel.
    
   
    GPPComplex expr0(0.00, 0.00);
    GPPComplex expr(0.5, 0.5);

    GPPComplex *acht_n1_loc = new GPPComplex[number_bands];
    GPPComplex *achtemp = new GPPComplex[nend-nstart];
    GPPComplex *asxtemp = new GPPComplex[nend-nstart];
    GPPComplex *aqsmtemp = new GPPComplex[number_bands*ncouls];
    GPPComplex *aqsntemp = new GPPComplex[number_bands*ncouls];
    GPPComplex *I_eps_array = new GPPComplex[ngpown*ncouls];
    GPPComplex *wtilde_array = new GPPComplex[ngpown*ncouls];
    GPPComplex *ssx_array = new GPPComplex[3];
    GPPComplex *ssxa = new GPPComplex[ncouls];
    GPPComplex achstemp;

    double *achtemp_re = new double[nend-nstart];
    double *achtemp_im = new double[nend-nstart];
                        
    double *vcoul = new double[ncouls];
    double wx_array[3];
    double occ=1.0;
    bool flag_occ;
    double achstemp_real = 0.00, achstemp_imag = 0.00;
    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;


   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j] = GPPComplex_mult(expr,(double)(i+j));
           aqsntemp[i*ncouls+j] = GPPComplex_mult(expr,(double)(i+j));
       }

   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = GPPComplex_mult(expr,(double)(i+j));
           wtilde_array[i*ncouls+j] = GPPComplex_mult(expr,(double)(i+j));
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0*i;


    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0; ig<ncouls; ++ig)
        indinv[ig] = ig;
        indinv[ncouls] = ncouls-1;

       for(int iw=nstart; iw<nend; ++iw)
       {
           asxtemp[iw] = expr0;
           achtemp_re[iw] = 0.00;
           achtemp_im[iw] = 0.00;
       }

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }


    auto startKernelTimer = std::chrono::high_resolution_clock::now();

   till_nvband(number_bands, nvband, ngpown, ncouls, inv_igp_index, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, asxtemp, vcoul); 

#pragma omp parallel for 
    for(int n1 = 0; n1<number_bands; ++n1) 
        reduce_achstemp(n1, number_bands, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul, numThreads);

    for(int n1 = 0; n1<number_bands; ++n1) 
       noFlagOCCSolver(n1, nvband, ngpown, ncouls, inv_igp_index, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im); 

    std::chrono::duration<double> elapsedKernelTime = std::chrono::high_resolution_clock::now() - startKernelTimer;


    printf(" \n Final achstemp\n");
    achstemp.print();

    printf("\n Final achtemp\n");

    for(int iw=nstart; iw<nend; ++iw)
    {
        GPPComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
        achtemp[iw].print();
    }

    std::chrono::duration<double> elapsed_totalTime = std::chrono::high_resolution_clock::now() - start_totalTime;
    cout << "********** Kernel Time Taken **********= " << elapsedKernelTime.count() << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsed_totalTime.count() << " secs" << endl;

    free(acht_n1_loc);
    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(asxtemp);
    free(vcoul);
    free(ssx_array);
    free(achtemp_re);
    free(achtemp_im);

    return 0;
}
