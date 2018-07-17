#include "GPPComplex.h"
#define nstart 0
#define nend 3

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

__device__ void d_ssxt_scht_solver(GPPComplex *wtilde_array,GPPComplex *I_eps_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex &ssxt, GPPComplex &scht, int my_igp, int ncouls, int n1, int ngpown, int ig, double wxt, int igp)
{
    GPPComplex ssxa(0.00, 0.00);
    GPPComplex scha(0.00, 0.00);
    GPPComplex wtilde = wtilde_array[my_igp*ncouls+ig];
    GPPComplex wtilde2 = GPPComplex(wtilde);
    GPPComplex Omega2 = d_GPPComplex_product(wtilde2, I_eps_array[my_igp*ncouls+ig]);
    GPPComplex mygpvar1 = d_GPPComplex_conj(aqsmtemp[n1*ncouls+igp]);
    GPPComplex mygpvar2 = aqsmtemp[n1*ncouls+igp];
    GPPComplex matngmatmgp = d_GPPComplex_product(aqsntemp[n1*ncouls+ig] , mygpvar1);

    GPPComplex expr0( 0.0 , 0.0);
    double ssxcutoff;
    double to1 = 1e-6;
    double sexcut = 4.0;
    double limitone = 1.0/(to1*4.0);
    double limittwo = pow(0.5,2);
    GPPComplex sch, ssx;

    GPPComplex wdiff = d_doubleMinusGPPComplex(wxt , wtilde);

    GPPComplex cden = wdiff;
    double rden = 1/d_GPPComplex_real(d_GPPComplex_product(cden , d_GPPComplex_conj(cden)));
    GPPComplex delw = d_GPPComplex_mult(d_GPPComplex_product(wtilde , d_GPPComplex_conj(cden)) , rden);
    double delwr = d_GPPComplex_real(d_GPPComplex_product(delw , d_GPPComplex_conj(delw)));
    double wdiffr = d_GPPComplex_real(d_GPPComplex_product(wdiff , d_GPPComplex_conj(wdiff)));

    if((wdiffr > limittwo) && (delwr < limitone))
    {
        sch = d_GPPComplex_product(delw , I_eps_array[my_igp*ncouls +ig]);
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

    ssxcutoff = sexcut*d_GPPComplex_abs(I_eps_array[my_igp*ncouls + ig]);
    if((d_GPPComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;
    ssxa = d_GPPComplex_product(matngmatmgp,ssx);
    scha = d_GPPComplex_product(matngmatmgp,sch);

    ssxt += ssxa;
    scht += scha;
}

__global__ void d_flagOCC_solver(GPPComplex *asxtemp, int *inv_igp_index, double *vcoul, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, double *wx_array, int nvband, int ncouls, int ngpown, int numThreadsPerBlock)
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

        for(int iw = 0; iw < 3; ++iw)
        {
           GPPComplex ssx(0.00, 0.00);
           GPPComplex sch(0.00, 0.00);
            int igp = inv_igp_index[my_igp];
            if(igp >= ncouls)
                igp = ncouls-1;

            GPPComplex ssxt(0.00, 0.00);
            GPPComplex scht(0.00, 0.00);


            for( int x = 0; x < loopOverncouls && threadIdx.x < numThreadsPerBlock ; ++x) 
            {
                int ig = x*numThreadsPerBlock + threadIdx.x;
                d_ssxt_scht_solver(wtilde_array, I_eps_array, aqsmtemp, aqsntemp, ssxt, scht, my_igp, ncouls, n1, ngpown, ig, wx_array[iw], igp); 
            }
            if(leftOverncouls)
            {
                int ig = loopOverncouls*numThreadsPerBlock + threadIdx.x;
                if(ig < ncouls)
                    d_ssxt_scht_solver(wtilde_array, I_eps_array, aqsmtemp, aqsntemp, ssxt, scht, my_igp, ncouls, n1, ngpown, ig, wx_array[iw], igp); 
            }

            ssx+= ssxt;
            sch+= d_GPPComplex_mult(scht, 0.5);
            asxtemp[iw] += d_GPPComplex_mult(d_GPPComplex_mult(ssx, occ) , vcoul[igp]);
        }
    }
}

void d_till_nvbandKernel(GPPComplex *asxtemp, int *inv_igp_index, double *vcoul, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, double *wx_array, int nvband, int ncouls, int ngpown)
{
    dim3 numBlocks(nvband, ngpown);
    int numThreadsPerBlock = 32;
    d_flagOCC_solver<<<numBlocks,numThreadsPerBlock>>>(asxtemp, inv_igp_index, vcoul, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, wx_array, nvband, ncouls, ngpown, numThreadsPerBlock);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

}

__device__ void d_achstempSolver(int number_bands, int *inv_igp_index, int ncouls, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, int ngpown, double *vcoul, double &achstemp_re, double &achstemp_im, int n1, int my_igp)
{
    GPPComplex schs(0.0, 0.0);
    GPPComplex schstemp(0.0, 0.0);
    double to1 = 1e-6;
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


__global__ void d_reduce_achstempKernel(int number_bands, int *inv_igp_index, int ncouls, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, int ngpown, double *vcoul, double *achstemp_re, double *achstemp_im, int numThreadsPerBlock)
{

    int n1 = blockIdx.x;
    if(n1 < number_bands)
    {
        int loopOverngpown = 1, leftOverngpown = 0;

        if(ngpown > numThreadsPerBlock)
        {
            loopOverngpown = ngpown / numThreadsPerBlock;
            leftOverngpown = ngpown % numThreadsPerBlock;
        }

        double achstemp_re_loc = 0.00, achstemp_im_loc = 0.00;
        for( int x = 0; x < loopOverngpown && threadIdx.x < numThreadsPerBlock ; ++x) 
        {
            int my_igp = x*numThreadsPerBlock + threadIdx.x;
            d_achstempSolver(number_bands, inv_igp_index, ncouls, aqsmtemp, aqsntemp, I_eps_array, ngpown, vcoul, achstemp_re_loc, achstemp_im_loc, n1, my_igp);

        }
        if(leftOverngpown)
        {
            int my_igp = loopOverngpown*numThreadsPerBlock + threadIdx.x;
            if(my_igp < ngpown)
            {
                double achstemp_re_loc = 0.00, achstemp_im_loc = 0.00;
                d_achstempSolver(number_bands, inv_igp_index, ncouls, aqsmtemp, aqsntemp, I_eps_array, ngpown, vcoul, achstemp_re_loc, achstemp_im_loc, n1, my_igp);

            }
        }
        atomicAdd(achstemp_re , achstemp_re_loc);
        atomicAdd(achstemp_im , achstemp_im_loc);
    }
}

__global__ void d_achtempSolver(int number_bands, int ngpown, int ncouls, GPPComplex *wtilde_array, int *inv_igp_index, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex *acht_n1_loc, double *vcoul, double *wx_array, int numThreadsPerBlock, double *achtemp_re, double *achtemp_im)
{
    int block_id = blockIdx.x;
    if(block_id == 0)
    {
        for(int n1 = 0; n1<number_bands; ++n1) 
        {
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
                        GPPComplex wdiff = d_doubleMinusGPPComplex(wx_array[iw], wtilde_array[my_igp*ncouls+ig]);
                        GPPComplex delw = d_GPPComplex_mult(d_GPPComplex_product(wtilde_array[my_igp*ncouls+ig] , d_GPPComplex_conj(wdiff)), 1/d_GPPComplex_real(d_GPPComplex_product(wdiff, d_GPPComplex_conj(wdiff)))); 
                        GPPComplex sch_array = d_GPPComplex_mult(d_GPPComplex_product(d_GPPComplex_product(d_GPPComplex_conj(aqsmtemp[n1*ncouls+igp]), aqsntemp[n1*ncouls+ig]), d_GPPComplex_product(delw , I_eps_array[my_igp*ncouls+ig])), 0.5*vcoul[igp]);
                        achtemp_re_loc[iw] += d_GPPComplex_real(sch_array);
                        achtemp_im_loc[iw] += d_GPPComplex_imag(sch_array);
                    }
                }
                for(int iw = nstart; iw < nend; ++iw)
                {
                    achtemp_re[iw] += achtemp_re_loc[iw];
                    achtemp_im[iw] += achtemp_im_loc[iw];
                }
    } //ngpown
        }

    }
//    int n1 = blockIdx.x;
//    if(n1 < number_bands)
//    {
//        double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
//        for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}
//        GPPComplex sch_array[nend-nstart];
//        if(threadIdx.x == 0)
//        {
//            for(int my_igp=0; my_igp<ngpown; ++my_igp)
//            {
//                int igp = inv_igp_index[my_igp];
//                if(igp >= ncouls)
//                    igp = ncouls-1;
//        
//        
//                for(int ig = 0; ig<ncouls; ++ig)
//                {
//                    for(int iw = nstart; iw < nend; ++iw)
//                    {
//                        GPPComplex wdiff = d_doubleMinusGPPComplex(wx_array[iw], wtilde_array[my_igp*ncouls+ig]);
//                        GPPComplex delw = d_GPPComplex_mult(d_GPPComplex_product(wtilde_array[my_igp*ncouls+ig] , d_GPPComplex_conj(wdiff)), 1/d_GPPComplex_real(d_GPPComplex_product(wdiff, d_GPPComplex_conj(wdiff)))); 
//                        sch_array[iw] = d_GPPComplex_mult(d_GPPComplex_product(d_GPPComplex_product(d_GPPComplex_conj(aqsmtemp[n1*ncouls+igp]), aqsntemp[n1*ncouls+ig]), d_GPPComplex_product(delw , I_eps_array[my_igp*ncouls+ig])), 0.5*vcoul[igp]);
//                        achtemp_re_loc[iw] += d_GPPComplex_real(sch_array[iw]);
//                        achtemp_im_loc[iw] += d_GPPComplex_imag(sch_array[iw]);
//                    }
//                }
//                acht_n1_loc[n1] += d_GPPComplex_mult(sch_array[2] , vcoul[igp]);
//            }
//        }
//
//        for(int iw=nstart; iw<nend; ++iw)
//        {
//            atomicAdd(&achtemp_re[iw] , achtemp_re_loc[iw]);
//            atomicAdd(&achtemp_im[iw] , achtemp_im_loc[iw]);
//        }
//
//    }
}

void d_reduce_achstemp(int number_bands, int *inv_igp_index, int ncouls, GPPComplex *aqsmtemp, GPPComplex *aqsntemp,  GPPComplex *I_eps_array, double *achstemp_re, double *achstemp_im, int ngpown, double *vcoul)
{
    int numBlocks = number_bands;
    int numThreadsPerBlock = 32;

    d_reduce_achstempKernel<<<numBlocks, numThreadsPerBlock>>>(number_bands, inv_igp_index, ncouls, aqsmtemp, aqsntemp, I_eps_array, ngpown, vcoul, achstemp_re, achstemp_im, numThreadsPerBlock);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

}

void d_achtemp_kernel(int number_bands, int ngpown, int ncouls, GPPComplex *wtilde_array, int *inv_igp_index, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex *acht_n1_loc, double *wx_array, double* vcoul, double *achtemp_re, double *achtemp_im)
{
    int numBlocks = number_bands;
    int numThreadsPerBlock = 32;

//        double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
//        for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}
//        GPPComplex sch_array[3];
//    for(int n1 = 0; n1<number_bands; ++n1) 
//    {
//
//            for(int my_igp=0; my_igp<ngpown; ++my_igp)
//            {
//                int igp = inv_igp_index[my_igp];
//                if(igp >= ncouls)
//                    igp = ncouls-1;
//        
//        
//                for(int ig = 0; ig<ncouls; ++ig)
//                {
//                    for(int iw = nstart; iw < nend; ++iw)
//                    {
//                        GPPComplex wdiff = doubleMinusGPPComplex(wx_array[iw], wtilde_array[my_igp*ncouls+ig]);
//                        GPPComplex delw = GPPComplex_mult(GPPComplex_product(wtilde_array[my_igp*ncouls+ig] , GPPComplex_conj(wdiff)), 1/GPPComplex_real(GPPComplex_product(wdiff, GPPComplex_conj(wdiff)))); 
//                        sch_array[iw] = GPPComplex_mult(GPPComplex_product(GPPComplex_product(GPPComplex_conj(aqsmtemp[n1*ncouls+igp]), aqsntemp[n1*ncouls+ig]), GPPComplex_product(delw , I_eps_array[my_igp*ncouls+ig])), 0.5*vcoul[igp]);
//                        achtemp_re_loc[iw] += GPPComplex_real(sch_array[iw]);
//                        achtemp_im_loc[iw] += GPPComplex_imag(sch_array[iw]);
//                    }
//                }
//                acht_n1_loc[n1] += GPPComplex_mult(sch_array[2] , vcoul[igp]);
//            }
//        }
//        for(int iw=nstart; iw<nend; ++iw)
//        {
//            achtemp_re[iw] += achtemp_re_loc[iw];
//            achtemp_im[iw] += achtemp_im_loc[iw];
//        }

    d_achtempSolver<<<1, numThreadsPerBlock>>>(number_bands, ngpown, ncouls, wtilde_array, inv_igp_index, aqsmtemp, aqsntemp, I_eps_array, acht_n1_loc, vcoul, wx_array, numThreadsPerBlock, achtemp_re, achtemp_im);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}
