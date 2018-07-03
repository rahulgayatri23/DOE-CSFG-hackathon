#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
#include <chrono>
#include <mpi.h>

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

void reduce_achstemp(int n1, int* inv_igp_index, int ncouls, std::complex<double> *aqsmtemp, std::complex<double> *aqsntemp, std::complex<double> *I_eps_array, std::complex<double>& achstemp, int ngpown, double* vcoul)
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
        int igp = inv_igp_index[my_igp];
        if(igp == ncouls)
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
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <matrix_divider> " << endl;
        exit (0);
    }

    int mpiSize = 1, mpiRank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int nodes_per_group = atoi(argv[4]);
    int nstart = 0, nend = 3;

    int npes = mpiSize; //Represents the number of ranks per node
    int ngpown = ncouls / (nodes_per_group * npes); //Number of gvectors per mpi task
    double e_lk = 10;
    double dw = 1;

    int tid, numThreads;
    double to1 = 1e-6;

    double gamma = 0.5;
    double sexcut = 4.0;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);

    double e_n1kq= 6.0; 

    //Printing out the params passed.
    if(mpiRank == 0)
    {
        std::cout << "******************Running pure MPI version of the code with : *************************" << std::endl;
        std::cout << "mpiSize = " << mpiSize << "\t number_bands = " << number_bands \
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
    }

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
    double *vcoul = new double[ncouls];


    std::complex<double> achstemp = std::complex<double>(0.0, 0.0);
    std::complex<double> ssx_array[3], \
        sch_array[3], \
        scht, ssxt, wtilde;

    std::complex<double> *ssxa = new std::complex<double> [ncouls];
    std::complex<double> *scha = new std::complex<double> [ncouls];

    double wxt;
    double occ=1.0;
    bool flag_occ;

    if(mpiRank == 0)
    {
        cout << "Size of wtilde_array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
        cout << "Size of aqsntemp = " << (ncouls*number_bands*2.0*8) / pow(1024,2) << " Mbytes" << endl;
        cout << "Size of I_eps_array array = " << (ncouls*ngpown*2.0*8) / pow(1024,2) << " Mbytes" << endl;
    }

   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsntemp[i*ncouls+j] = ((double)(i+j))*expr;
           aqsmtemp[i*ncouls+j] = ((double)(i+j))*expr;
       }


   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           int global_i = ngpown*mpiRank + i;
           I_eps_array[i*ncouls+j] = ((double)(global_i+j))*expr;
           wtilde_array[i*ncouls+j] = ((double)(global_i+j))*expr;
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0*i;


    for(int ig=0; ig < ngpown; ++ig)
    {
       int global_ig = ngpown*mpiRank + ig;
        inv_igp_index[ig] = (global_ig+1) * ncouls / (ngpown*npes);
    }

       for(int iw=0; iw<3; ++iw)
           achtemp[iw] = expr0;

    auto startTimer = std::chrono::high_resolution_clock::now();

    for(int n1 = 0; n1<number_bands; ++n1) // This for loop at the end cheddam
    {
        flag_occ = n1 < nvband;

        reduce_achstemp(n1, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, ngpown, vcoul);

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(abs(wx_array[iw]) < to1) wx_array[iw] = to1;
        }

        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            //JRD changedthis
            int igp = inv_igp_index[my_igp];
            if(igp == ncouls)
                igp = ncouls-1;
            int igmax;

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
                achtemp[iw] += sch_array[iw] * vcoul[igp];


            acht_n1_loc[n1] += sch_array[2] * vcoul[igp];

            } //for the if-loop to avoid break inside an openmp pragma statment
        } //ngpown
    } //number_bands n1
    std::chrono::duration<double> elapsedTimer = std::chrono::high_resolution_clock::now() - startTimer;

    double achtemp_re[3], achtemp_im[3];
    for(int iw=nstart; iw<nend; ++iw)
    {
        achtemp_re[iw] = real(achtemp[iw]);
        achtemp_im[iw] = imag(achtemp[iw]);
    }


    if(mpiRank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, achtemp_re, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE, achtemp_im, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Reduce(achtemp_re, achtemp_re, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(achtemp_im, achtemp_im, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    


    if(mpiRank == 0)
    {
        for(int iw=nstart; iw<nend; ++iw)
            cout << "achtemp[" << iw << "] = (" << achtemp_re[iw] << ", " << achtemp_im[iw] << ") " << endl;

        cout << "********** Time Taken **********= " << elapsedTimer.count() << " secs" << endl;
    }


    free(acht_n1_loc);
    free(wtilde_array);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(inv_igp_index);
    free(vcoul);

    MPI_Finalize();

    return 0;
}

//Almost done code
