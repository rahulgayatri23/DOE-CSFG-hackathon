#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <chrono>

using namespace std;


void ssxt_scht_solver(double wxt, int igp, int my_igp, int ig, std::complex<double> wtilde, std::complex<double> wtilde2, std::complex<double> Omega2, std::complex<double> matngmatmgp, std::complex<double> matngpmatmg, std::complex<double> mygpvar1, std::complex<double> mygpvar2, std::complex<double>& ssxa, std::complex<double>& scha, std::complex<double> I_eps_array_igp_myIgp)
{
    std::complex<double> expr0( 0.0 , 0.0);
    double delw2, scha_mult, ssxcutoff;
    double to1 = 1e-6;
    double sexcut = 4.0;
    double gamma = 0.5;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    std::complex<double> sch, ssx;

    std::complex<double> wdiff = wxt - wtilde;

    std::complex<double> cden = wdiff;
    double rden = 1/real(cden * conj(cden));
    std::complex<double> delw = wtilde * conj(cden) * rden;
    double delwr = real(delw * conj(delw));
    double wdiffr = real(wdiff * conj(wdiff));

    if((wdiffr > limittwo) && (delwr < limitone))
    {
        sch = delw * I_eps_array_igp_myIgp;
        cden = pow(wxt,2);
        rden = real(cden * conj(cden));
        rden = 1.00 / rden;
        ssx = Omega2 * conj(cden) * rden;
    }
    else if (delwr > to1)
    {
        sch = expr0;
        cden = (double) 4.00 * wtilde2 * (delw + (double)0.50);
        rden = real(cden * conj(cden));
        rden = 1.00/rden;
        ssx = -Omega2 * conj(cden) * rden * delw;
    }
    else
    {
        sch = expr0;
        ssx = expr0;
    }

    ssxcutoff = sexcut*abs(I_eps_array_igp_myIgp);
    if((abs(ssx) > ssxcutoff) && (abs(wxt) < 0.00)) ssx = 0.00;

    ssxa = matngmatmgp*ssx;
    scha = matngmatmgp*sch;
}

void reduce_achstemp(int n1, int* inv_igp_index, int ncouls, std::complex<double> *aqsmtemp, std::complex<double> *aqsntemp, std::complex<double> *I_eps_array, std::complex<double>& achstemp,  int* indinv, int ngpown, double* vcoul)
{
    double to1 = 1e-6;
    int igmax;
    std::complex<double> schstemp(0.0, 0.0);;
    for(int my_igp = 0; my_igp< ngpown; my_igp++)
    {
        std::complex<double> schs(0.0, 0.0);
        std::complex<double> matngmatmgp(0.0, 0.0);
        std::complex<double> matngpmatmg(0.0, 0.0);
        std::complex<double> halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
        int igmax;
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        if(indigp == ncouls)
            igp = ncouls-1;

        if(!(igp > ncouls || igp < 0)){

        igmax = ncouls;

        std::complex<double> mygpvar1 = std::conj(aqsmtemp[n1*ncouls+igp]);
        std::complex<double> mygpvar2 = aqsntemp[n1*ncouls+igp];

            schs = -I_eps_array[my_igp*ncouls+igp];
            matngmatmgp = aqsntemp[n1*ncouls+igp] * mygpvar1;

            if(abs(schs) > to1)
                schstemp = schstemp + matngmatmgp * schs;
        }
        else
        {
            for(int ig=1; ig<igmax; ++ig)
                schstemp = schstemp - aqsntemp[n1*ncouls+igp] * I_eps_array[my_igp*ncouls+ig] * mygpvar1;
        }
        achstemp += schstemp * vcoul[igp] *(double) 0.5;
    }
}



void flagOCC_solver(double wxt, std::complex<double> *wtilde_array, int my_igp, int n1, std::complex<double> *aqsmtemp, std::complex<double> *aqsntemp, std::complex<double> *I_eps_array, std::complex<double> &ssxt, std::complex<double> &scht, int igmax, int ncouls, int igp, std::complex<double> *ssxa, std::complex<double>* scha)
{
    std::complex<double> matngmatmgp = std::complex<double>(0.0, 0.0);
    std::complex<double> matngpmatmg = std::complex<double>(0.0, 0.0);
    for(int ig=0; ig<igmax; ++ig)
    {
        std::complex<double> wtilde = wtilde_array[my_igp*ncouls+ig];
        std::complex<double> wtilde2 = std::pow(wtilde,2);
        std::complex<double> Omega2 = wtilde2*I_eps_array[my_igp*ncouls+ig];
        std::complex<double> mygpvar1 = std::conj(aqsmtemp[n1*ncouls+igp]);
        std::complex<double> mygpvar2 = aqsmtemp[n1*ncouls+igp];
        std::complex<double> matngmatmgp = aqsntemp[n1*ncouls+ig] * mygpvar1;
        if(ig != igp) matngpmatmg = std::conj(aqsmtemp[n1*ncouls+ig]) * mygpvar2;

        ssxt_scht_solver(wxt, igp, my_igp, ig, wtilde, wtilde2, Omega2, matngmatmgp, matngpmatmg, mygpvar1, mygpvar2, ssxa[ig], scha[ig], I_eps_array[my_igp*ncouls+ig]); 
        ssxt += ssxa[ig];
        scht += scha[ig];
    }
}

void noflagOCC_solver(double wxt, std::complex<double> *wtilde_array, int my_igp, int n1, std::complex<double> *aqsmtemp, std::complex<double> *aqsntemp, std::complex<double> *I_eps_array, std::complex<double> &ssxt, std::complex<double> &scht, int igmax, int ncouls, int igp, std::complex<double> *scha)
{
    double to1 = 1e-6;
    double sexcut = 4.0;
    double gamma = 0.5;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    std::complex<double> mygpvar1 = std::conj(aqsmtemp[n1*ncouls+igp]);
    std::complex<double> scht_loc(0.00, 0.00);
    
//#pragma simd
//#pragma ivdep
    for(int ig = 0; ig<ncouls; ++ig)
    {
        std::complex<double> wdiff = wxt - wtilde_array[my_igp*ncouls+ig];
        double wdiffr = real(wdiff * conj(wdiff));
        double rden = 1/wdiffr;

        std::complex<double> delw = wtilde_array[my_igp*ncouls+ig] * conj(wdiff) *rden; //*rden
        double delwr = real(delw * conj(delw));

        scht_loc += mygpvar1 * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] ;
    }

    scht = scht_loc;
    
//    for(int ig = 0; ig<ncouls; ++ig)
//    {
//        std::complex<double> wdiff = wxt - wtilde_array[my_igp*ncouls+ig];
//        double wdiffr = real(wdiff * conj(wdiff));
//        double rden = 1/wdiffr;
//
//        std::complex<double> delw = wtilde_array[my_igp*ncouls+ig] * conj(wdiff) *rden; //*rden
//        double delwr = real(delw * conj(delw));
//
//
//        if((wdiffr > limittwo) && (delwr < limitone))
//            scha[ig] = mygpvar1 * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] ;
//
//    }
//    for(int ig = 0; ig<ncouls; ++ig)
//        scht += scha[ig];
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
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

    double e_n1kq= 6.0; //This in the fortran code is derived through the double dimenrsion array ekq whose 2nd dimension is 1 and all the elements in the array have the same value
//    MyAllocator<64> alloc;


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

    std::complex<double> expr0( 0.0 , 0.0);
    std::complex<double> expr( 0.5 , 0.5);
    std::complex<double> achtemp[3];
    std::complex<double> asxtemp[3];
    double wx_array[3];

    // Memory allocation
    std::complex<double> *acht_n1_loc = new std::complex<double> [number_bands];
    std::complex<double> *aqsmtemp = new std::complex<double> [number_bands*ncouls];
    std::complex<double> *aqsntemp = new std::complex<double> [number_bands*ncouls];
    std::complex<double> *I_eps_array = new std::complex<double> [ngpown*ncouls];
    std::complex<double> *wtilde_array = new std::complex<double> [ngpown*ncouls];

    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls];
    double *vcoul = new double[ncouls];
    std::complex<double> *achtempThread = new std::complex<double> [numThreads * 3];


    std::complex<double> achstemp = std::complex<double>(0.0, 0.0);
    std::complex<double> ssx_array[3], \
        sch_array[3], \
        scht, ssxt, wtilde;

    std::complex<double> *ssxa = new std::complex<double> [ncouls];
    std::complex<double> *scha = new std::complex<double> [ncouls];

    double wxt;
    double occ=1.0;
    bool flag_occ;

    cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;

   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsntemp[i*ncouls+j] = expr;
           aqsmtemp[i*ncouls+j] = expr;
       }


   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = expr;
           wtilde_array[i*ncouls+j] = expr;
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0;


    for(int ig=0, tmp=1; ig < ngpown; ++ig,tmp++)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0, tmp=1; ig<ncouls; ++ig,tmp++)
        indinv[ig] = ig;

   for(int n=0; n<numThreads; ++n)
   {
       for(int iw=0; iw<3; ++iw)
       {
           achtempThread[n*3 + iw] = expr0;
           achtemp[iw] = expr0;
       }
   }

    auto startTimer = std::chrono::high_resolution_clock::now();

    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
        flag_occ = n1 < nvband;

        reduce_achstemp(n1, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul);

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;
        }

#pragma omp parallel for default(shared) private(scht, sch_array, ssx_array, tid) firstprivate(ngpown, ncouls) schedule(dynamic) 
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            if(indigp == ncouls)
                igp = ncouls-1;
            int igmax;
            tid = omp_get_thread_num();

            if(!(igp > ncouls || igp < 0)) {
                igmax = ncouls;

            for(int i=0; i<3; i++)
            {
                ssx_array[i] = expr0;
                sch_array[i] = expr0;
            }

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; iw++)
                {
                    scht = ssxt = expr0;
                    wxt = wx_array[iw];
                    flagOCC_solver(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, igmax, ncouls, igp, ssxa, scha);

                    ssx_array[iw] += ssxt;
                    sch_array[iw] +=(double) 0.5*scht;
                }
            }
            else
            {
                for(int iw=nstart; iw<nend; ++iw)
                {
                        scht = ssxt = expr0;
                        wxt = wx_array[iw];

                        noflagOCC_solver(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, igmax, ncouls, igp, scha);

#pragma omp critical
                        sch_array[iw] +=(double) 0.5*scht;
                }
            }

            if(flag_occ)
            {
                for(int iw=nstart; iw<nend; ++iw)
                {
                    asxtemp[iw] += ssx_array[iw] * occ * vcoul[igp];
                }
            }

            for(int iw=nstart; iw<nend; ++iw)
                achtempThread[tid*3 + iw] += sch_array[iw] * vcoul[igp];

            acht_n1_loc[n1] += sch_array[2] * vcoul[igp];

            } //for the if-loop to avoid break inside an openmp pragma statment
        }
    }
    std::chrono::duration<double> elapsedTimer = std::chrono::high_resolution_clock::now() - startTimer;

    for(int n = 0; n < numThreads; ++n)
    {
        for(int iw = 0; iw < 3; ++iw)
            achtemp[iw] += achtempThread[n*3 + iw];
    }


    double end_time = omp_get_wtime(); //End timing here

    for(int iw=nstart; iw<nend; ++iw)
        cout << "achtemp[" << iw << "] = " << std::setprecision(15) << achtemp[iw] << endl;

    cout << "********** Time Taken **********= " << elapsedTimer.count() << " secs" << endl;

    free(acht_n1_loc);
    free(wtilde_array);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(inv_igp_index);
    free(indinv);
    free(vcoul);
    free(achtempThread);

    return 0;
}

//Almost done code
