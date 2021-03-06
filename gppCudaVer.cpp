#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
//#include <chrono>
#include "GPPComplex.h"
#include <sys/time.h>

using namespace std;

#define nstart 0
#define nend 3

//Outputs are ssxa and scha, rest of the passed parameters are the inputs
void ssxt_scht_solver(double wxt, int igp, int my_igp, int ig, GPPComplex wtilde, GPPComplex wtilde2, GPPComplex Omega2, GPPComplex matngmatmgp, GPPComplex matngpmatmg, GPPComplex mygpvar1, GPPComplex mygpvar2, GPPComplex& ssxa, GPPComplex& scha, GPPComplex I_eps_array_igp_myIgp)
{
    GPPComplex expr0( 0.0 , 0.0);
    double ssxcutoff;
    double to1 = 1e-6;
    double sexcut = 4.0;
    double gamma = 0.5;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    GPPComplex sch(0.00, 0.00);
    GPPComplex ssx(0.00, 0.00);

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
//        cden = (double) 4.00 * wtilde2 * (delw + (double)0.50);
        rden = GPPComplex_real(GPPComplex_product(cden , GPPComplex_conj(cden)));
        rden = 1.00/rden;
        ssx = GPPComplex_product(GPPComplex_product(-Omega2 , GPPComplex_conj(cden)), GPPComplex_mult(delw, rden));
    }
    else
    {
        sch = expr0;
        ssx = expr0;
    }

    ssxcutoff = GPPComplex_abs(I_eps_array_igp_myIgp) * sexcut;
    if((GPPComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

    ssxa += GPPComplex_product(matngmatmgp , ssx);
    scha += GPPComplex_product(matngmatmgp , sch);
}

//This function writes its results to achstemp, rest of the parameters are its inputs.
void reduce_achstemp(int n1, int* inv_igp_index, int ncouls, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex& achstemp, int ngpown, double* vcoul)
{
    double to1 = 1e-6;
    for(int my_igp = 0; my_igp< ngpown; my_igp++)
    {
        GPPComplex schstemp(0.0, 0.0);
        GPPComplex schs(0.0, 0.0);
        GPPComplex matngmatmgp(0.0, 0.0);
        GPPComplex matngpmatmg(0.0, 0.0);
        GPPComplex mygpvar1(0.00, 0.00);
        GPPComplex mygpvar2(0.00, 0.00);
        int igp = inv_igp_index[my_igp];
        if(igp >= ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0))
        {

            mygpvar1 = GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
            mygpvar2 = aqsntemp[n1*ncouls+igp];
            GPPComplex schs_pos = I_eps_array[my_igp*ncouls+igp];

            schs = -schs_pos;
            matngmatmgp = GPPComplex_product(aqsntemp[n1*ncouls+igp] , mygpvar1);

            if(GPPComplex_abs(schs) > to1)
                GPPComplex_fma(schstemp, matngmatmgp, schs);
        }
        else
        {
            for(int ig=1; ig<ncouls; ++ig)
                GPPComplex_fms(schstemp, GPPComplex_product(aqsntemp[n1*ncouls+ig], I_eps_array[my_igp*ncouls+ig]), mygpvar1);
        }
        achstemp += GPPComplex_mult(schstemp , vcoul[igp] *(double) 0.5);
    }
}



//Performs the calculation for the first nvband iterations.
//Outputs are ssxt and scht, rest of the passed parameters are the inputs
void flagOCC_solver(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht, int ncouls, int igp)
{
    GPPComplex matngmatmgp = GPPComplex(0.0, 0.0);
    GPPComplex matngpmatmg = GPPComplex(0.0, 0.0);
    GPPComplex expr0(0.00, 0.00);
    for(int ig=0; ig<ncouls; ++ig)
    {
        GPPComplex ssxa(0.00, 0.00);
        GPPComplex scha(0.00, 0.00);
        GPPComplex wtilde = wtilde_array[my_igp*ncouls+ig];
        GPPComplex wtilde2 = GPPComplex_square(wtilde);
        GPPComplex Omega2 = GPPComplex_product(wtilde2,I_eps_array[my_igp*ncouls+ig]);
        GPPComplex mygpvar1 = GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
        GPPComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
        GPPComplex matngmatmgp = GPPComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
        if(ig != igp) matngpmatmg = GPPComplex_product(GPPComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);

        ssxt_scht_solver(wxt, igp, my_igp, ig, wtilde, wtilde2, Omega2, matngmatmgp, matngpmatmg, mygpvar1, mygpvar2, ssxa, scha, I_eps_array[my_igp*ncouls+ig]); 
        ssxt += ssxa;
        scht += scha;
    }
}

//Outputs is scht, rest of the passed parameters are the inputs
void noflagOCC_iter(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht, int ncouls, int igp)
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
        GPPComplex wdiff = doubleMinusGPPComplex(wxt , wtilde_array[my_igp*ncouls+ig]);
        double wdiffr = GPPComplex_real(GPPComplex_product(wdiff , GPPComplex_conj(wdiff)));
        double rden = 1/wdiffr;

        GPPComplex delw = GPPComplex_mult(GPPComplex_product(wtilde_array[my_igp*ncouls+ig] , GPPComplex_conj(wdiff)) ,rden); 
        double delwr = GPPComplex_real(GPPComplex_product(delw , GPPComplex_conj(delw)));

        scht_loc += GPPComplex_product(GPPComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]) , GPPComplex_product(delw , I_eps_array[my_igp*ncouls+ig])) ;
    }

    scht = scht_loc;
}

//This function calculates the first nvband iterations of the outermost loop
//output achtemp, asxtemp, acht_n1_loc
void till_nvband(int nvband, int ngpown, int ncouls, int *inv_igp_index, double *wx_array, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex *asxtemp, double *vcoul, GPPComplex *acht_n1_loc, double *achtemp_re, double *achtemp_im, const double occ)
{

    for(int n1 = 0; n1<nvband; ++n1) 
    {
        GPPComplex scht(0.00, 0.00);
        GPPComplex ssxt(0.00, 0.00);
        GPPComplex expr0(0.00, 0.00);

        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            GPPComplex ssx_array[nend-nstart];
            GPPComplex sch_array[nend-nstart];
            int igp = inv_igp_index[my_igp];
            if(igp >= ncouls)
                igp = ncouls-1;

            for(int iw=nstart; iw<nend; iw++)
            {
                ssx_array[iw] = expr0;
                sch_array[iw] = expr0;
                scht = ssxt = expr0;
                double wxt = wx_array[iw];
                flagOCC_solver(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp);
                ssx_array[iw] += ssxt;
                sch_array[iw] +=GPPComplex_mult(scht, 0.5);
                
                asxtemp[iw] += GPPComplex_mult(ssx_array[iw] , occ * vcoul[igp]);//Store output of the first nvband iterations.
            }

            for(int iw=nstart; iw<nend; ++iw)
            {
                achtemp_re[iw] += GPPComplex_real(GPPComplex_mult(sch_array[iw] , vcoul[igp]));
                achtemp_im[iw] += GPPComplex_imag(GPPComplex_mult(sch_array[iw] , vcoul[igp]));
            }

            acht_n1_loc[n1] += GPPComplex_mult(sch_array[2] , vcoul[igp]);
        }
    }
}



void noFlagOCCSolver(int n1, int nvband, int ngpown, int ncouls, int *inv_igp_index, double *wx_array, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, GPPComplex *acht_n1_loc)
{
    GPPComplex expr0(0.00, 0.00);
    GPPComplex ssxt(0.00, 0.00);
    GPPComplex scht(0.00, 0.00);

    for(int my_igp=0; my_igp<ngpown; ++my_igp)
    {
        GPPComplex sch_array[nend-nstart];
        int igp = inv_igp_index[my_igp];
        if(igp >= ncouls)
            igp = ncouls-1;

        for(int iw=nstart; iw<nend; ++iw)
        {
            sch_array[iw] = expr0;
            scht = ssxt = expr0;
            double wxt = wx_array[iw];
            noflagOCC_iter(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp);

            sch_array[iw] +=GPPComplex_mult(scht, 0.5);
        }

        for(int iw=nstart; iw<nend; ++iw)
        {
            achtemp_re[iw] += GPPComplex_real(GPPComplex_mult(sch_array[iw] , vcoul[igp]));
            achtemp_im[iw] += GPPComplex_imag(GPPComplex_mult(sch_array[iw] , vcoul[igp]));
        }

        acht_n1_loc[n1] += GPPComplex_mult(sch_array[2] , vcoul[igp]);
    }
}

int main(int argc, char** argv)
{

//The input to the executable needs 4 arguments.
    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <matrix_divider> " << endl;
        exit (0);
    }

//Input parameters stored in these variables.
    const int number_bands = atoi(argv[1]);
    const int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int nodes_per_group = atoi(argv[4]);

//Constants that will be used later
    const int npes = 1; 
    const int ngpown = ncouls / (nodes_per_group * npes); 
    const double e_lk = 10;
    const double dw = 1;
    const double to1 = 1e-6;
    const double gamma = 0.5;
    const double sexcut = 4.0;
    const double limitone = 1.0/(to1*4.0);
    const double limittwo = pow(0.5,2);
    const double e_n1kq= 6.0; 
    const double occ=1.0;

    //Printing out the params passed.
    std::cout << "**************************** Cuda  with GPPComplex class for GPP code ************************* " << std::endl;
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

    // Memory allocation of input data structures.
    // Two dimensional arrays from theory have been initialized as a single dimension in m*n format for performance.
    GPPComplex *acht_n1_loc = new GPPComplex [number_bands];
    GPPComplex *aqsmtemp = new GPPComplex [number_bands*ncouls];
    GPPComplex *aqsntemp = new GPPComplex [number_bands*ncouls];
    GPPComplex *I_eps_array = new GPPComplex [ngpown*ncouls];
    GPPComplex *wtilde_array = new GPPComplex [ngpown*ncouls];
    int *inv_igp_index = new int[ngpown];
    double *vcoul = new double[ncouls];
    double wx_array[nend-nstart];

    //arrays that will be later used to store the output results
    GPPComplex achtemp[nend-nstart]; 
    GPPComplex asxtemp[nend-nstart];
    GPPComplex achstemp = GPPComplex(0.0, 0.0);

//Divide the achtemp into real and imaginary parts to avoid critical and use atomic
    double achtemp_re[nend-nstart];
    double achtemp_im[nend-nstart];

    //Allocat same data structures on Device so as to optimize on data-transfer
    GPPComplex *d_acht_n1_loc, *d_aqsmtemp, *d_aqsntemp, *d_I_eps_array, *d_wtilde_array, *d_asxtemp;
    double *d_achtemp_re, *d_achtemp_im, *d_vcoul, *d_wx_array, *d_achstemp_re, *d_achstemp_im, \
       achstemp_re, achstemp_im;
    int *d_inv_igp_index;

    CudaSafeCall(cudaMalloc((void**) &d_acht_n1_loc, number_bands*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsntemp, number_bands*ncouls*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_I_eps_array, ngpown*ncouls*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_wtilde_array, ngpown*ncouls*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_wx_array, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_inv_igp_index, ngpown*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_vcoul, ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_re, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_im, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_asxtemp, (nend-nstart)*sizeof(GPPComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_achstemp_re, sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achstemp_im, sizeof(double)));

    //Data structures that store intermediete results
    GPPComplex scht, ssxt;

    //Printing the size of each of the input data structures.
    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

    //Some expressions declared to be used later in the initialization.
    GPPComplex expr0( 0.0 , 0.0);
    GPPComplex expr( 0.5 , 0.5);

//Initializing the data structures
   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsntemp[i*ncouls+j] = GPPComplex_mult(expr, (double)(i+j));
           aqsmtemp[i*ncouls+j] = GPPComplex_mult(expr, (double)(i+j));
       }


   for(int i=0; i<ngpown; i++)
   {
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = GPPComplex_mult(expr, (double)(i+j));
           wtilde_array[i*ncouls+j] = GPPComplex_mult(expr, (double)(i+j));
       }

        inv_igp_index[i] = (i+1) * ncouls / ngpown;
   }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0*i;


    for(int iw=nstart; iw<nend; ++iw)
    {
        achtemp[iw] = expr0;
        asxtemp[iw] = expr0;
        achtemp_re[iw] = 0.00;
        achtemp_im[iw] = 0.00;

        wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
        if(wx_array[iw] < to1) wx_array[iw] = to1;
    }

    //Update data structures on device
    CudaSafeCall(cudaMemcpy(d_acht_n1_loc, acht_n1_loc, number_bands*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_I_eps_array, I_eps_array, ngpown*ncouls*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_wtilde_array, wtilde_array, ngpown*ncouls*sizeof(GPPComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_inv_igp_index, inv_igp_index, ngpown*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_wx_array, wx_array, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achtemp_re, achtemp_re, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achtemp_im, achtemp_im, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));


    //Start the timer before the work begins.
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);

    //reduce_achstemp is computed on the host device itself. Can use the CPU threads to compute this.
    for(int n1 = 0; n1<number_bands; ++n1) 
        reduce_achstemp(n1, inv_igp_index, ncouls, aqsmtemp, aqsntemp, I_eps_array, achstemp, ngpown, vcoul);

    //This routine generates a cuda-kernel to compute 0-nvband outer iteration 
    d_till_nvband(nvband, ngpown, ncouls, d_inv_igp_index, d_wx_array, d_wtilde_array, d_aqsmtemp, d_aqsntemp, d_I_eps_array, d_asxtemp, d_vcoul, d_acht_n1_loc, d_achtemp_re, d_achtemp_im, occ); 

    //nvband-number_bands iterations are computed inside another cuda-kernel generated in the following routine.
   d_noFlagOCCSolver(number_bands, nvband, ngpown, ncouls, d_inv_igp_index, d_wx_array, d_wtilde_array, d_aqsmtemp, d_aqsntemp, d_I_eps_array, d_vcoul, d_achtemp_re, d_achtemp_im, d_acht_n1_loc); 

    //Wait for the cuda kernels to finish their computation
    cudaDeviceSynchronize();

    //Copy the results back, divide the final achtemp array into real and imaginary double arrays to use atomics.
    CudaSafeCall(cudaMemcpy(achtemp_im, d_achtemp_im, 3*sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(achtemp_re, d_achtemp_re, 3*sizeof(double), cudaMemcpyDeviceToHost));

    //Time Taken
    gettimeofday(&endTimer, NULL);
    double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);


    for(int iw=nstart; iw<nend; ++iw)
    {
        GPPComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
        achtemp[iw].print();
    }

    cout << "********** Time Taken **********= " << elapsedTimer << " secs" << endl;

    //Free the allocated memory
    free(acht_n1_loc);
    free(wtilde_array);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(inv_igp_index);
    free(vcoul);

    return 0;
}
