#ifndef __GPPCOMPLEX
#define __GPPCOMPLEX

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <complex>
#include <omp.h>
#include <ctime>
#include <chrono>


#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NumBandsKernel 1
#define NgpownKernel 0
#define NumBandsNgpownKernel 0
#define NgpownNcoulsKernel 0

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
        file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed. - Rahul - commented the below deviceSynchronize
//    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}


class GPPComplex : public double2{

    private : 

public:
__host__ __device__ explicit GPPComplex () {
    x = 0.00;
    y = 0.00;
}


__host__ __device__ explicit GPPComplex(const double& a, const double& b) {
    x = a;
    y = b;
}

__host__ __device__ GPPComplex(const GPPComplex& src) {
    x = src.x;
    y = src.y;
}

__host__ __device__ GPPComplex& operator =(const GPPComplex& src) {
    x = src.x;
    y = src.y;

    return *this;
}

__host__ __device__ GPPComplex& operator +=(const GPPComplex& src) {
    x = src.x + this->x;
    y = src.y + this->y;

    return *this;
}

__host__ __device__ GPPComplex& operator -=(const GPPComplex& src) {
    x = src.x - this->x;
    y = src.y - this->y;

    return *this;
}

__host__ __device__ GPPComplex& operator -() {
    x = -this->x;
    y = -this->y;

    return *this;
}

__host__ __device__ GPPComplex& operator ~() {
    return *this;
}

void print() const {
    printf("( %f, %f) ", this->x, this->y);
    printf("\n");
}

double abs(const GPPComplex& src) {

    double re_this = src.x * src.x;
    double im_this = src.y * src.y;

    double result = (re_this+im_this);
    return result;
}

double get_real() const
{
    return this->x;
}

double get_imag() const
{
    return this->y;
}

void set_real(double val)
{
    this->x = val;
}

void set_imag(double val) 
{
    this->y = val;
}

    friend inline GPPComplex GPPComplex_square(GPPComplex& src) ;
    friend inline GPPComplex GPPComplex_conj(const GPPComplex& src) ;
    friend inline GPPComplex GPPComplex_product(const GPPComplex& a, const GPPComplex& b) ;
    friend inline double GPPComplex_abs(const GPPComplex& src) ;
    friend inline GPPComplex GPPComplex_mult(GPPComplex& a, double b, double c) ;
    friend inline GPPComplex GPPComplex_mult(const GPPComplex& a, double b) ;
    friend inline void GPPComplex_fma(GPPComplex& a, const GPPComplex& b, const GPPComplex& c) ;
    friend inline void GPPComplex_fms(GPPComplex& a, const GPPComplex& b, const GPPComplex& c) ;
    friend inline GPPComplex doubleMinusGPPComplex(const double &a, GPPComplex& src) ;
    friend inline GPPComplex doublePlusGPPComplex(double a, GPPComplex& src) ;
    friend inline double GPPComplex_real( const GPPComplex& src) ;
    friend inline double GPPComplex_imag( const GPPComplex& src) ;

    
//Device Functions 
    friend __device__ const GPPComplex d_GPPComplex_square(GPPComplex& src) ;
    friend __device__ const GPPComplex d_GPPComplex_conj(const GPPComplex& src) ;
    friend __device__ const GPPComplex d_GPPComplex_product(const GPPComplex& a, const GPPComplex& b) ;
    friend __device__ double d_GPPComplex_abs(const GPPComplex& src) ;
    friend __device__ const GPPComplex d_GPPComplex_mult(GPPComplex& a, double b, double c) ;
    friend __device__ const GPPComplex d_GPPComplex_mult(const GPPComplex& a, double b) ;
    friend __device__ void d_GPPComplex_fma(GPPComplex& a, const GPPComplex& b, const GPPComplex& c) ;
    friend __device__ void d_GPPComplex_fms(GPPComplex& a, const GPPComplex& b, const GPPComplex& c) ;
    friend __device__ GPPComplex d_doubleMinusGPPComplex(const double &a, GPPComplex& src) ;
    friend __device__ const GPPComplex d_doublePlusGPPComplex(double a, GPPComplex& src) ;
    friend __device__ double d_GPPComplex_real( const GPPComplex& src) ;
    friend __device__ double d_GPPComplex_imag( const GPPComplex& src) ;
    friend __device__ void d_GPPComplex_plusEquals( GPPComplex& a, const GPPComplex & b); 
    friend __device__ void d_GPPComplex_Equals( GPPComplex& a, const GPPComplex & b); 
    friend __device__ void d_print( const GPPComplex& src) ;
    friend __device__ void ncoulsKernel(GPPComplex& mygpvar1, GPPComplex& wdiff, GPPComplex& aqsntemp_index, GPPComplex& wtilde_array_index, GPPComplex& I_eps_array_index, double vcoul_igp, double& achtemp_re_loc, double& achtemp_im_loc);
    friend __device__ void ncoulsKernel(GPPComplex& mygpvar1, double vcoul_igp, double& achtemp_re_loc, double& achtemp_im_loc);

};
//Inline functions have to be defined in the same file as the declaration

/*
 * Return the square of a complex number 
 */
GPPComplex GPPComplex_square(GPPComplex& src) {
    double re_this = src.x ;
    double im_this = src.y ;

    GPPComplex result(re_this*re_this - im_this*im_this, 2*re_this*im_this);

    return result;
}

/*
 * Return the conjugate of a complex number 
 */
GPPComplex GPPComplex_conj(const GPPComplex& src) {

    double re_this = src.x;
    double im_this = -1 * src.y;

    GPPComplex result(re_this, im_this);
    return result;
}


/*
 * Return the product of 2 complex numbers 
 */
GPPComplex GPPComplex_product(const GPPComplex& a, const GPPComplex& b) {

    double re_this = a.x * b.x - a.y*b.y ;
    double im_this = a.x * b.y + a.y*b.x ;

    GPPComplex result(re_this, im_this);
    return result;
}

/*
 * Return the absolute of a complex number 
 */
double GPPComplex_abs(const GPPComplex& src) {
    double re_this = src.x * src.x;
    double im_this = src.y * src.y;

    double result = (re_this+im_this);
    return result;
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
GPPComplex GPPComplex_mult(GPPComplex& a, double b, double c) {

    GPPComplex result(a.x * b * c, a.y * b * c);
    return result;

}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
GPPComplex GPPComplex_mult(const GPPComplex& a, double b) {

   GPPComplex result(a.x*b, a.y*b);
   return result;

}

/*
 * Return the complex number a += b * c  
 */
void GPPComplex_fma(GPPComplex& a, const GPPComplex& b, const GPPComplex& c) {
    double re_this = b.x * c.x - b.y*c.y ;
    double im_this = b.x * c.y + b.y*c.x ;

    GPPComplex mult_result(re_this, im_this);

    a.x += mult_result.x;
    a.y += mult_result.y;
}

/*
 * Return the complex number a -= b * c  
 */
void GPPComplex_fms(GPPComplex& a, const GPPComplex& b, const GPPComplex& c) {
    double re_this = b.x * c.x - b.y*c.y ;
    double im_this = b.x * c.y + b.y*c.x ;

    GPPComplex mult_result(re_this, im_this);

    a.x -= mult_result.x;
    a.y -= mult_result.y;
}


GPPComplex doubleMinusGPPComplex(const double &a, GPPComplex& src) {
    GPPComplex result(a - src.x, 0 - src.y);
    return result;
}

GPPComplex doublePlusGPPComplex(double a, GPPComplex& src) {
    GPPComplex result(a + src.x, 0 + src.y);
    return result;
}

double GPPComplex_real( const GPPComplex& src) {
    return src.x;
}

double GPPComplex_imag( const GPPComplex& src) {
    return src.y;
}

void gppKernelGPU( GPPComplex *wtilde_array, GPPComplex *aqsntemp, GPPComplex* aqsmtemp, GPPComplex *I_eps_array, int ncouls, int ngpown, int number_bands, double* wx_array, double *achtemp_re, double *achtemp_im, double *vcoul, int nstart, int nend, int* indinv, int* inv_igp_index);

void till_nvbandKernel(GPPComplex *asxtemp, int *inv_igp_index, double *vcoul, GPPComplex *wtilde_array, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, double* wx_array, int nvband, int ncouls, int ngpown);

void d_reduce_achstemp(int number_bands, int *inv_igp_index, int ncouls, GPPComplex *aqsmtemp, GPPComplex *aqsntemp,  GPPComplex *I_eps_array, double *achstemp_re, double *achstemp_im, int ngpown, double *vcoul);

void d_achtemp_kernel(int number_bands, int ngpown, int ncouls, GPPComplex *wtilde_array, int *inv_igp_index, GPPComplex *aqsmtemp, GPPComplex *aqsntemp, GPPComplex *I_eps_array, GPPComplex *acht_n1_loc, double *wx_array, double* vcoul, double *achtemp_re, double *achtemp_im);
#endif
