#include "GPPComplex.h"
#define nstart 0
#define nend 3

// Atomic add operation for double
#if defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 600
#define atomicAdd2 atomicAdd
#else
__device__ double atomicAdd2( double *address, double val )
{
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old     = atomicCAS( address_as_ull, assumed, __double_as_longlong( val + __longlong_as_double( assumed ) ) );
    } while ( assumed != old );
    return __longlong_as_double( old );
}
#endif

/*
 * Return the square of a complex number 
 */
__device__ const inline GPPComplex d_GPPComplex_square(GPPComplex& src) {
    return GPPComplex(src.x*src.x - src.y*src.y, 2*src.x*src.y);
}

/*
 * Return the conjugate of a complex number 
 */
__device__ const inline GPPComplex d_GPPComplex_conj(const GPPComplex& src) {
return GPPComplex(src.x, -src.y);
}


/*
 * Return the product of 2 complex numbers 
 */
__device__ const inline GPPComplex d_GPPComplex_product(const GPPComplex& a, const GPPComplex& b) {
    return GPPComplex(a.x * b.x - a.y*b.y, a.x * b.y + a.y*b.x);
}


/*
 * Return the absolute of a complex number 
 */
__device__ inline double d_GPPComplex_abs(const GPPComplex& src) {
    return sqrt(src.x * src.x + src.y * src.y);
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
__device__ const inline GPPComplex d_GPPComplex_mult(GPPComplex& a, double b, double c) {
    return GPPComplex(a.x * b * c, a.y * b * c);
}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
__device__ const inline GPPComplex d_GPPComplex_mult(const GPPComplex& a, double b) {
   return GPPComplex(a.x*b, a.y*b);

}

/*
 * Return the complex number a += b * c  
 */
__device__ inline void d_GPPComplex_fma(GPPComplex& a, const GPPComplex& b, const GPPComplex& c) {
    a.x += b.x * c.x - b.y*c.y ;
    a.y += b.x * c.y + b.y*c.x ;
}

/*
 * Return the complex number a -= b * c  
 */
__device__ inline void d_GPPComplex_fms(GPPComplex& a, const GPPComplex& b, const GPPComplex& c) {
    a.x -= b.x * c.x - b.y*c.y ;
    a.y -= b.x * c.y + b.y*c.x ;
}


__device__ inline GPPComplex d_doubleMinusGPPComplex(const double &a, GPPComplex& src) {
    return GPPComplex(a-src.x, -src.y);
}

__device__ const inline GPPComplex d_doublePlusGPPComplex(double a, GPPComplex& src) {
    return GPPComplex(a+src.x, src.y);
}

__device__ inline double d_GPPComplex_real( const GPPComplex& src) {
    return src.x;
}

__device__ inline double d_GPPComplex_imag( const GPPComplex& src) {
    return src.y;
}

__device__ inline void d_GPPComplex_plusEquals( GPPComplex& a, const GPPComplex & b) {
    a.x += b.x;
    a.y += b.y;
}

__device__ void inline d_GPPComplex_Equals( GPPComplex& a, const GPPComplex & b) {
    a.x = b.x;
    a.y = b.y;
}

__device__ void d_print( const GPPComplex& a) {
    printf("( %f, %f) ", a.x, a.y);
    printf("\n");
}

__device__ void d_ssxt_scht_solver(double wxt, int igp, int my_igp, int ig, GPPComplex wtilde, GPPComplex wtilde2, GPPComplex Omega2, GPPComplex matngmatmgp, GPPComplex matngpmatmg, GPPComplex mygpvar1, GPPComplex mygpvar2, GPPComplex& ssxa, GPPComplex& scha, GPPComplex I_eps_array_igp_myIgp)
{
    GPPComplex expr0( 0.0 , 0.0);
    double ssxcutoff;
    double to1 = 1e-6;
    double sexcut = 4.0;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    GPPComplex sch(0.00, 0.00);
    GPPComplex ssx(0.00, 0.00);

    GPPComplex wdiff = d_doubleMinusGPPComplex(wxt , wtilde);

    GPPComplex cden = wdiff;
    double rden = 1/d_GPPComplex_real(d_GPPComplex_product(cden , d_GPPComplex_conj(cden)));
    GPPComplex delw = d_GPPComplex_mult(d_GPPComplex_product(wtilde , d_GPPComplex_conj(cden)) , rden);
    double delwr = d_GPPComplex_real(d_GPPComplex_product(delw , d_GPPComplex_conj(delw)));
    double wdiffr = d_GPPComplex_real(d_GPPComplex_product(wdiff , d_GPPComplex_conj(wdiff)));

    if((wdiffr > limittwo) && (delwr < limitone))
    {
        sch = d_GPPComplex_product(delw , I_eps_array_igp_myIgp);
        double cden = std::pow(wxt,2);
        rden = std::pow(cden,2);
        rden = 1.00 / rden;
        ssx = d_GPPComplex_mult(Omega2 , cden , rden);
    }
    else if (delwr > to1)
    {
        sch = expr0;
        cden = d_GPPComplex_mult(d_GPPComplex_product(wtilde2, d_doublePlusGPPComplex((double)0.50, delw)), 4.00);
        rden = d_GPPComplex_real(d_GPPComplex_product(cden , d_GPPComplex_conj(cden)));
        rden = 1.00/rden;
        ssx = d_GPPComplex_product(d_GPPComplex_product(-Omega2 , d_GPPComplex_conj(cden)), d_GPPComplex_mult(delw, rden));
    }
    else
    {
        sch = expr0;
        ssx = expr0;
    }

    ssxcutoff = d_GPPComplex_abs(I_eps_array_igp_myIgp) * sexcut;
    if((d_GPPComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

    ssxa += d_GPPComplex_product(matngmatmgp , ssx);
    scha += d_GPPComplex_product(matngmatmgp , sch);
}

__global__ void d_flagOCC_solver(GPPComplex *asxtemp, int *inv_igp_index, double *vcoul, GPPComplex *acht_n1_loc, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, double *wx_array, int nvband, int ncouls, int ngpown, int numThreadsPerBlock, double *achtemp_re, double *achtemp_im)
{
    int n1 = blockIdx.x ;
    int my_igp = blockIdx.y;

    if(n1 < nvband && my_igp < ngpown)
    {
        double occ = 1.00;
        int loopOverncouls = 1, leftOverncouls = 0;

        if(ncouls > numThreadsPerBlock)
        {
            loopOverncouls = ncouls / numThreadsPerBlock;
            leftOverncouls = ncouls % numThreadsPerBlock;
        }

        GPPComplex scht(0.00, 0.00);
        GPPComplex ssxt(0.00, 0.00);
        GPPComplex expr0(0.00, 0.00);
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

            GPPComplex matngmatmgp = GPPComplex(0.0, 0.0);
            GPPComplex matngpmatmg = GPPComplex(0.0, 0.0);

            for( int x = 0; x < loopOverncouls && threadIdx.x < numThreadsPerBlock ; ++x) 
            {
                int ig = x*numThreadsPerBlock + threadIdx.x;
                GPPComplex ssxa(0.00, 0.00);
                GPPComplex scha(0.00, 0.00);
                GPPComplex wtilde = wtilde_array[my_igp*ncouls+ig];
                GPPComplex wtilde2 = d_GPPComplex_square(wtilde);
                GPPComplex Omega2 = d_GPPComplex_product(wtilde2,I_eps_array[my_igp*ncouls+ig]);
                GPPComplex mygpvar1 = d_GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
                GPPComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
                GPPComplex matngmatmgp = d_GPPComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
                if(ig != igp) matngpmatmg = d_GPPComplex_product(d_GPPComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);

                d_ssxt_scht_solver(wxt, igp, my_igp, ig, wtilde, wtilde2, Omega2, matngmatmgp, matngpmatmg, mygpvar1, mygpvar2, ssxa, scha, I_eps_array[my_igp*ncouls+ig]); 
                ssxt += ssxa;
                scht += scha;
            }
            if(leftOverncouls)
            {
                int ig = loopOverncouls*numThreadsPerBlock + threadIdx.x;
                if(ig < ncouls)
                {
                    GPPComplex ssxa(0.00, 0.00);
                    GPPComplex scha(0.00, 0.00);
                    GPPComplex wtilde = wtilde_array[my_igp*ncouls+ig];
                    GPPComplex wtilde2 = d_GPPComplex_square(wtilde);
                    GPPComplex Omega2 = d_GPPComplex_product(wtilde2,I_eps_array[my_igp*ncouls+ig]);
                    GPPComplex mygpvar1 = d_GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
                    GPPComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
                    GPPComplex matngmatmgp = d_GPPComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);
                    if(ig != igp) matngpmatmg = d_GPPComplex_product(d_GPPComplex_conj(aqsmtemp[n1*ncouls+ig]) , mygpvar2);

                    d_ssxt_scht_solver(wxt, igp, my_igp, ig, wtilde, wtilde2, Omega2, matngmatmgp, matngpmatmg, mygpvar1, mygpvar2, ssxa, scha, I_eps_array[my_igp*ncouls+ig]); 
                    ssxt += ssxa;
                    scht += scha;
                }
            }

            ssx_array[iw] += ssxt;
            sch_array[iw] +=d_GPPComplex_mult(scht, 0.5);
            
            asxtemp[iw] += d_GPPComplex_mult(ssx_array[iw] , occ * vcoul[igp]);//Store output of the first nvband iterations.
        }
        for(int iw=nstart; iw<nend; ++iw)
        {
            atomicAdd2(&achtemp_re[iw] , d_GPPComplex_real(d_GPPComplex_mult(sch_array[iw] , vcoul[igp])));
            atomicAdd2(&achtemp_im[iw] , d_GPPComplex_imag(d_GPPComplex_mult(sch_array[iw] , vcoul[igp])));
        }

            acht_n1_loc[n1] += d_GPPComplex_mult(sch_array[2] , vcoul[igp]);
    }

}


__device__ void d_achstempSolver(int number_bands, int *inv_igp_index, int ncouls, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, int ngpown, double *vcoul, double &achstemp_re, double &achstemp_im, int n1, int my_igp)
{
    GPPComplex schs(0.0, 0.0);
    GPPComplex schstemp(0.0, 0.0);
    GPPComplex matngmatmgp(0.0, 0.0);
    GPPComplex matngpmatmg(0.0, 0.0);
    GPPComplex halfinvwtilde, delw, ssx, sch, wdiff, cden , eden, mygpvar1, mygpvar2;
    int igp = inv_igp_index[my_igp];
   if(igp >= ncouls)
       igp = ncouls-1;

    if(!(igp > ncouls || igp < 0)){

    GPPComplex mygpvar1 = d_GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
    GPPComplex mygpvar2 = aqsntemp[n1*ncouls+igp];

        schs = -I_eps_array[my_igp*ncouls+igp];
        matngmatmgp = d_GPPComplex_product(aqsntemp[n1*ncouls+igp] , mygpvar1);

//        if(d_GPPComplex_abs(schs) > to1)
//            d_GPPComplex_fma(schstemp, matngmatmgp, schs);
    }
    else
    {
        for(int ig=1; ig<ncouls; ++ig)
        {
            GPPComplex mult_result(d_GPPComplex_product(I_eps_array[my_igp*ncouls+ig] , mygpvar1));
            d_GPPComplex_fms(schstemp,aqsntemp[n1*ncouls+igp], mult_result); 
        }
    }
    achstemp_re += d_GPPComplex_real(d_GPPComplex_mult(schstemp , vcoul[igp] * 0.5));
    achstemp_im += d_GPPComplex_imag(d_GPPComplex_mult(schstemp , vcoul[igp] * 0.5));
}

__device__ void d_noflagOCC_solver(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht, int ncouls, int igp)
{
    GPPComplex mygpvar1 = d_GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
    GPPComplex scht_loc(0.00, 0.00);
    
    for(int ig = 0; ig<ncouls; ++ig)
    {
        GPPComplex wdiff = d_doubleMinusGPPComplex(wxt, wtilde_array[my_igp*ncouls+ig]);
        double wdiffr = d_GPPComplex_real(d_GPPComplex_product(wdiff , d_GPPComplex_conj(wdiff)));
        double rden = 1/wdiffr;

        GPPComplex delw = d_GPPComplex_mult(d_GPPComplex_product(wtilde_array[my_igp*ncouls+ig] , d_GPPComplex_conj(wdiff)) ,rden); 
        double delwr = d_GPPComplex_real(d_GPPComplex_product(delw , d_GPPComplex_conj(delw)));

        scht_loc += d_GPPComplex_product(d_GPPComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]) , d_GPPComplex_product(delw , I_eps_array[my_igp*ncouls+ig])) ;
    }

    scht = scht_loc;
}

__device__ void d_noflagOCC_iter(double wxt, GPPComplex *wtilde_array, int my_igp, int n1, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex &ssxt, GPPComplex &scht, int ncouls, int igp)
{
    double limittwo = pow(0.5,2);
    GPPComplex mygpvar1 = d_GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
    GPPComplex scht_loc(0.00, 0.00);
    
    for(int ig = 0; ig<ncouls; ++ig)
    {
        GPPComplex wdiff = d_doubleMinusGPPComplex(wxt , wtilde_array[my_igp*ncouls+ig]);
        double wdiffr = d_GPPComplex_real(d_GPPComplex_product(wdiff , d_GPPComplex_conj(wdiff)));
        double rden = 1/wdiffr;

        GPPComplex delw = d_GPPComplex_mult(d_GPPComplex_product(wtilde_array[my_igp*ncouls+ig] , d_GPPComplex_conj(wdiff)) ,rden); 
        double delwr = d_GPPComplex_real(d_GPPComplex_product(delw , d_GPPComplex_conj(delw)));

        scht_loc += d_GPPComplex_product(d_GPPComplex_product(mygpvar1 , aqsntemp[n1*ncouls+ig]) , d_GPPComplex_product(delw , I_eps_array[my_igp*ncouls+ig])) ;
    }

    scht = scht_loc;
}

__global__ void d_achtempSolver(int number_bands, int nvband, int ngpown, int ncouls, int *inv_igp_index, double *wx_array, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, GPPComplex *acht_n1_loc, int numThreadsPerBlock)
{
    int n1 = blockIdx.x;
    if(n1 >= nvband && n1 < number_bands)
    {

        GPPComplex expr0(0.00, 0.00);
        GPPComplex ssxt(0.00, 0.00);
        GPPComplex scht(0.00, 0.00);
        GPPComplex sch_array[nend-nstart];
        double achtemp_re_loc[nend-nstart];
        double achtemp_im_loc[nend-nstart];
        int loopOverngpown = 1, leftOverngpown = 0;

        for(int iw = nstart; iw < nend; ++iw)
        {
            sch_array[iw] = expr0;
            achtemp_re_loc[iw] = 0.00;
            achtemp_im_loc[iw] = 0.00;
        }

        if(ncouls > numThreadsPerBlock)
        {
            loopOverngpown = ngpown / numThreadsPerBlock;
            leftOverngpown = ncouls % numThreadsPerBlock;
        }
        for( int x = 0; x < loopOverngpown && threadIdx.x < numThreadsPerBlock ; ++x) 
        {
            int my_igp = x*numThreadsPerBlock + threadIdx.x;
            int igp = inv_igp_index[my_igp];
            if(igp >= ncouls)
                igp = ncouls-1;

            for(int iw=nstart; iw<nend; ++iw)
            {
                sch_array[iw] = expr0;
                scht = ssxt = expr0;
                double wxt = wx_array[iw];
                d_noflagOCC_iter(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp);

                sch_array[iw] +=d_GPPComplex_mult(scht, 0.5);
                achtemp_re_loc[iw] += d_GPPComplex_real(d_GPPComplex_mult(sch_array[iw] , vcoul[igp]));
                achtemp_im_loc[iw] += d_GPPComplex_imag(d_GPPComplex_mult(sch_array[iw] , vcoul[igp]));
            }
            acht_n1_loc[n1] += d_GPPComplex_mult(sch_array[2] , vcoul[igp]);
        }
        if(leftOverngpown)
        {
            int my_igp = loopOverngpown*numThreadsPerBlock + threadIdx.x;
            GPPComplex sch_array[nend-nstart];
            int igp = inv_igp_index[my_igp];
            if(igp >= ncouls)
                igp = ncouls-1;

            for(int iw=nstart; iw<nend; ++iw)
            {
                sch_array[iw] = expr0;
                scht = ssxt = expr0;
                double wxt = wx_array[iw];
                d_noflagOCC_iter(wxt, wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp);

                sch_array[iw] +=d_GPPComplex_mult(scht, 0.5);
                achtemp_re_loc[iw] += d_GPPComplex_real(d_GPPComplex_mult(sch_array[iw] , vcoul[igp]));
                achtemp_im_loc[iw] += d_GPPComplex_imag(d_GPPComplex_mult(sch_array[iw] , vcoul[igp]));
            }
            acht_n1_loc[n1] += d_GPPComplex_mult(sch_array[2] , vcoul[igp]);
        }

        for(int iw=nstart; iw<nend; ++iw)
        {
            atomicAdd2(&achtemp_re[iw], achtemp_re_loc[iw]);
            atomicAdd2(&achtemp_im[iw], achtemp_im_loc[iw]);
        }

    }
}

//The routine launches a cuda kernel that runs the first nvband iterations of the outermost loop
void d_till_nvband(int nvband, int ngpown, int ncouls, int *inv_igp_index, double *wx_array, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex *asxtemp, double *vcoul, GPPComplex *acht_n1_loc, double *achtemp_re, double *achtemp_im, const double occ)
{
    dim3 numBlocks(nvband, ngpown);
    int numThreadsPerBlock = 32;

//Launches a two dimensional cuda-kernel with the first-dimension of nvband and 2nd dimension as ngpown.
//Each grid block is generated with numThreadsPerBlock (32) threads.
    d_flagOCC_solver<<<numBlocks,numThreadsPerBlock>>>(asxtemp, inv_igp_index, vcoul, acht_n1_loc, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, wx_array, nvband, ncouls, ngpown, numThreadsPerBlock, achtemp_re, achtemp_im);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

}

//Launches a single dimension cuda-kernel with the number_bands gridSize.
//Each grid block is generated with numThreadsPerBlock (32) threads.
void d_noFlagOCCSolver(int number_bands, int nvband, int ngpown, int ncouls, int *inv_igp_index, double *wx_array, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, GPPComplex *acht_n1_loc)
{
    int numBlocks = number_bands;
    int numThreadsPerBlock = 2;

//Launches a two dimensional cuda-kernel with the first-dimension of nvband and 2nd dimension as ngpown.
    d_achtempSolver<<<numBlocks, numThreadsPerBlock>>>(number_bands, nvband, ngpown, ncouls, inv_igp_index, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im, acht_n1_loc, numThreadsPerBlock);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}
