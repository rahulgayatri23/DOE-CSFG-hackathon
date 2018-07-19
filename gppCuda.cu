#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

#define CUDA_CHECK()                                          \
    do {                                                      \
        if ( cudaPeekAtLastError() != cudaSuccess ) {         \
            cudaError_t error = cudaGetLastError();           \
            printf( "cuda error: %i\n", error );              \
            printf( "   %s\n", cudaGetErrorString( error ) ); \
            printf( "   line: %i\n", (int) __LINE__ );        \
            printf( "   file: %s\n", __FILE__ );              \
            exit( -1 );                                       \
        }                                                     \
    } while ( 0 )


// Atomic add operation for double
#if defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 600
#define atomicAdd2 atomicAdd
#else
__device__ double atomicAdd2( double *address, double val )
{
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old             = *address_as_ull, assumed;
    do {
        assumed = old;
        old     = atomicCAS( address_as_ull, assumed,
            __double_as_longlong( val + __longlong_as_double( assumed ) ) );
    } while ( assumed != old );
    return __longlong_as_double( old );
}
#endif


template<class TYPE>
TYPE *allocate( size_t count )
{
    TYPE *x;
    cudaMallocManaged( (void **) &x, count * sizeof( TYPE ) );
    memset( x, 0, count * sizeof( TYPE ) );
    return x;
}


class Complex
{
public:
    __device__ __host__ Complex( double r = 0, double i = 0 ) : re( r ), im( i ) {}
    __device__ __host__ Complex &operator+=( const Complex &x ) {
        re += x.re;
        im += x.im;
        return *this;
    }
    __device__ __host__ double &real() { return re; }
    __device__ __host__ double &imag() { return im; }
    __device__ __host__ friend Complex operator*( const Complex &a, const Complex &b ) {
        return Complex( a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re );
    }
    __device__ __host__ friend Complex operator*( const Complex &a, double b ) {
        return Complex( a.re * b, a.im * b );
    }
    __device__ __host__ friend Complex operator*( double a, const Complex &b ) {
        return Complex( a * b.re, a * b.im );
    }
    __device__ __host__ friend Complex operator+( const Complex &a, const Complex &b ) {
        return Complex( a.re + b.re, a.im + b.im );
    }
    __device__ __host__ friend Complex operator-( const Complex &a, const Complex &b ) {
        return Complex( a.re - b.re, a.im - b.im );
    }
    __device__ __host__ friend Complex operator-( const Complex &x ) { return Complex( -x.re, -x.im ); }
    __device__ __host__ friend Complex conj( const Complex &x ) { return Complex( x.re, -x.im ); }
    __device__ __host__ friend double real( const Complex &x ) { return x.re; }
    __device__ __host__ friend double imag( const Complex &x ) { return x.im; }
    __device__ __host__ friend double abs( const Complex &x ) {
        return sqrt( x.re * x.re + x.im * x.im );
    }
    __device__ __host__ friend double abs2( const Complex &x ) {
        return x.re * x.re + x.im * x.im;
    }
    friend ostream &operator<<( ostream &os, const Complex &x ) {
        os << "(" << x.re << "," << x.im << ")";
        return os;
    }

private:
    double re;
    double im;
};


__device__ __host__ inline void ssxt_scht_solver( double wxt, int igp, int my_igp, int ig,
    const Complex& wtilde, const Complex& Omega2, const Complex& matngmatmgp,
    Complex &ssxa, Complex &scha,
    const Complex& I_eps_array_igp_myIgp )
{
    const double to1      = 1e-6;
    const double limitone = 1.0 / ( to1 * 4.0 );
    const double limittwo = 0.25;   // 0.5^2

    Complex wdiff = wxt - wtilde;
    Complex cden( wdiff );
    double rden   = 1 / abs2( cden );
    Complex delw  = wtilde * conj( cden ) * rden;
    double delwr  = abs2( delw  );
    double wdiffr = abs2( wdiff );

    Complex sch, ssx;
    if ( ( wdiffr > limittwo ) && ( delwr < limitone ) ) {
        sch  = delw * I_eps_array_igp_myIgp;
        cden = wxt * wxt;
        rden = abs2( cden );
        rden = 1.00 / rden;
        ssx  = Omega2 * conj( cden ) * rden;
    } else if ( delwr > to1 ) {
        sch  = Complex( 0.0, 0.0 );
        cden = 4.0 * wtilde * wtilde * ( delw + 0.50 );
        rden = abs2( cden );
        rden = 1.00 / rden;
        ssx  = -Omega2 * conj( cden ) * rden * delw;
    } else {
        sch = ssx = Complex( 0.0, 0.0 );
    }
    ssxa = matngmatmgp * ssx;
    scha = matngmatmgp * sch;
}

/*__device__ __host__ inline void reduce_achstemp( int n1, const int *inv_igp_index, int ncouls,
    const Complex *aqsmtemp, const Complex *aqsntemp, const Complex *I_eps_array, Complex &achstemp,
    int ngpown, const double *vcoul )
{
    double to1 = 1e-6;
    Complex schstemp( 0.0, 0.0 );
    for ( int my_igp = 0; my_igp < ngpown; my_igp++ ) {
        Complex mygpvar1, mygpvar2;
        int igp = inv_igp_index[my_igp];
        if ( igp >= ncouls )
            igp = ncouls - 1;

        if ( !( igp > ncouls || igp < 0 ) ) {

            Complex mygpvar1 = conj( aqsmtemp[n1 * ncouls + igp] );
            Complex mygpvar2 = aqsntemp[n1 * ncouls + igp];

            Complex schs = -I_eps_array[my_igp * ncouls + igp];
            Complex matngmatmgp = aqsntemp[n1 * ncouls + igp] * mygpvar1;

            if ( abs( schs ) > to1 )
                schstemp = schstemp + matngmatmgp * schs;
        } else {
            for ( int ig = 1; ig < ncouls; ++ig )
                schstemp = schstemp - aqsntemp[n1 * ncouls + igp] *
                                          I_eps_array[my_igp * ncouls + ig] * mygpvar1;
        }
        achstemp += 0.5 * schstemp * vcoul[igp];
    }
}*/


__device__ __host__ inline void flagOCC_solver( double wxt, const Complex *wtilde_array, int my_igp,
    int n1, const Complex *aqsmtemp, const Complex *aqsntemp, const Complex *I_eps_array,
    Complex &ssxt, Complex &scht, int ncouls, int igp )
{
    Complex ssxa, scha;
    for ( int ig = 0; ig < ncouls; ++ig ) {
        Complex wtilde      = wtilde_array[my_igp * ncouls + ig];
        Complex Omega2      = wtilde * wtilde * I_eps_array[my_igp * ncouls + ig];
        Complex mygpvar1    = conj( aqsmtemp[n1 * ncouls + igp] );
        Complex matngmatmgp = aqsntemp[n1 * ncouls + ig] * mygpvar1;
        ssxt_scht_solver( wxt, igp, my_igp, ig, wtilde, Omega2, matngmatmgp,
            ssxa, scha, I_eps_array[my_igp * ncouls + ig] );
        ssxt += ssxa;
        scht += scha;
    }
}

__device__ __host__ inline void noflagOCC_solver( double wxt, const Complex *wtilde_array,
    int my_igp, int n1, const Complex *aqsmtemp, const Complex *aqsntemp,
    const Complex *I_eps_array, Complex &scht, int ncouls, int igp )
{
    Complex mygpvar1 = conj( aqsmtemp[n1 * ncouls + igp] );
    Complex scht_loc( 0.00, 0.00 );
    for ( int ig = 0; ig < ncouls; ++ig ) {
        Complex wdiff = wxt - wtilde_array[my_igp * ncouls + ig];
        double wdiffr = abs2( wdiff );
        double rden   = 1.0 / wdiffr;
        Complex delw  = wtilde_array[my_igp * ncouls + ig] * conj( wdiff ) * rden; //*rden
        double delwr  = abs2( delw );
        scht_loc     +=
            mygpvar1 * aqsntemp[n1 * ncouls + ig] * delw * I_eps_array[my_igp * ncouls + ig];
    }
    scht = scht_loc;
}


__device__ inline void calculate( int n1, int my_igp, int nvband, const int *inv_igp_index,
    int ncouls, const double *wx_array, const Complex *aqsmtemp,
    const Complex *aqsntemp, const Complex *I_eps_array, const Complex *wtilde_array,
    Complex *achtemp, Complex *asxtemp, const double *vcoul )
{
    const int nstart = 0, nend = 3;
    const double occ = 1.0;
    bool flag_occ = n1 < nvband;
    int igp       = inv_igp_index[my_igp];
    if ( igp >= ncouls )
        igp = ncouls - 1;

    if ( flag_occ ) {
        Complex scht, ssxt;
        for ( int iw = nstart; iw < nend; iw++ ) {
            flagOCC_solver( wx_array[iw], wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt,
                scht, ncouls, igp );
            asxtemp[iw] += ssxt * ( occ * vcoul[igp] );
            achtemp[iw] += (double) 0.5 * scht * vcoul[igp];
        }
    } else {
        Complex scht;
        for ( int iw = nstart; iw < nend; ++iw ) {
            noflagOCC_solver( wx_array[iw], wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array,
                scht, ncouls, igp );
            achtemp[iw] += (double) 0.5 * ( scht * vcoul[igp] );
        }
    }
}


// Get the globally unique thread id
__device__ int getGlobalIdx3D()
{
    int blockId  = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * ( blockDim.x * blockDim.y * blockDim.z ) +
                   ( threadIdx.z * ( blockDim.x * blockDim.y ) ) + ( threadIdx.y * blockDim.x ) +
                   threadIdx.x;
    return threadId;
}


// Compute kernel
__global__ void call_compute( int number_bands, int ngpown, int nvband, const int *inv_igp_index,
    int ncouls, const double *wx_array, const Complex *aqsmtemp,
    const Complex *aqsntemp, const Complex *I_eps_array, const Complex *wtilde_array,
    Complex *achtemp, Complex *asxtemp, const double *vcoul )
{
    int tid    = getGlobalIdx3D();
    int n1     = tid % number_bands;
    int my_igp = tid / number_bands;
    // Perform the local computations
    Complex achtemp2[3], asxtemp2[3];
    if ( n1 < number_bands && my_igp < ngpown )
        calculate( n1, my_igp, nvband, inv_igp_index, ncouls, wx_array, aqsmtemp,
            aqsntemp, I_eps_array, wtilde_array, achtemp2, asxtemp2, vcoul );
    // Perform the reduce
    for ( int i = 0; i < 3; i++ ) {
        atomicAdd2( &achtemp[i].real(), achtemp2[i].real() );
        atomicAdd2( &achtemp[i].imag(), achtemp2[i].imag() );
        atomicAdd2( &asxtemp[i].real(), asxtemp2[i].real() );
        atomicAdd2( &asxtemp[i].imag(), asxtemp2[i].imag() );
    }
}


int main( int argc, char **argv )
{
    if ( argc != 5 ) {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <matrix_divider>\n";
        exit( 0 );
    }
    int number_bands    = atoi( argv[1] );
    int nvband          = atoi( argv[2] );
    int ncouls          = atoi( argv[3] );
    int nodes_per_group = atoi( argv[4] );

    int npes   = 1;                                   // Represents the number of ranks per node
    int ngpown = ncouls / ( nodes_per_group * npes ); // Number of gvectors per mpi task

    double e_lk      = 10;
    double dw        = 1;
    const int nstart = 0, nend = 3;

    double to1      = 1e-6;
    double gamma    = 0.5;
    double sexcut   = 4.0;
    double limitone = 1.0 / ( to1 * 4.0 );
    double limittwo = 0.25;
    double e_n1kq   = 6.0;


    // Printing out the params passed.
    std::cout << "number_bands = " << number_bands << "\t nvband = " << nvband
              << "\t ncouls = " << ncouls << "\t nodes_per_group  = " << nodes_per_group
              << "\t ngpown = " << ngpown << "\t nend = " << nend << "\t nstart = " << nstart
              << "\t gamma = " << gamma << "\t sexcut = " << sexcut << "\t limitone = " << limitone
              << "\t limittwo = " << limittwo << endl;

    const Complex expr0( 0.0, 0.0 );
    const Complex expr( 0.5, 0.5 );

    // Memory allocation
    Complex *aqsmtemp     = allocate<Complex>( number_bands * ncouls );
    Complex *aqsntemp     = allocate<Complex>( number_bands * ncouls );
    Complex *I_eps_array  = allocate<Complex>( ngpown * ncouls );
    Complex *wtilde_array = allocate<Complex>( ngpown * ncouls );
    int *inv_igp_index    = allocate<int>( ngpown );
    double *vcoul         = allocate<double>( ncouls );
    Complex *achtemp      = allocate<Complex>( 3 );
    Complex *asxtemp      = allocate<Complex>( 3 );
    double *wx_array      = allocate<double>( 3 );
    CUDA_CHECK();

    cout << "Size of wtilde_array = " << ( ncouls * ngpown * 2.0 * 8 ) / 1048576 << " Mbytes" << endl;
    cout << "Size of aqsntemp = " << ( ncouls * number_bands * 2.0 * 8 ) / 1048576 << " Mbytes" << endl;
    cout << "Size of I_eps_array array = " << ( ncouls * ngpown * 2.0 * 8 ) / 1048576 << " Mbytes" << endl;

    for ( int i = 0; i < number_bands; i++ )
        for ( int j = 0; j < ncouls; j++ ) {
            aqsntemp[i * ncouls + j] = ( (double) ( i + j ) ) * expr;
            aqsmtemp[i * ncouls + j] = ( (double) ( i + j ) ) * expr;
        }


    for ( int i = 0; i < ngpown; i++ )
        for ( int j = 0; j < ncouls; j++ ) {
            I_eps_array[i * ncouls + j]  = ( (double) ( i + j ) ) * expr;
            wtilde_array[i * ncouls + j] = ( (double) ( i + j ) ) * expr;
        }

    for ( int i = 0; i < ncouls; i++ )
        vcoul[i] = 1.0 * i;


    for ( int ig = 0; ig < ngpown; ++ig )
        inv_igp_index[ig] = ( ig + 1 ) * ncouls / ngpown;

    for ( int iw = nstart; iw < nend; ++iw ) {
        achtemp[iw]  = expr0;
        wx_array[iw] = e_lk - e_n1kq + dw * ( ( iw + 1 ) - 2 );
        if ( abs( wx_array[iw] ) < to1 )
            wx_array[iw] = to1;
    }

    // Prefetch memory for read only variables
    int device = -1;
    cudaGetDevice( &device );
    cudaMemPrefetchAsync( aqsmtemp, number_bands * ncouls * sizeof( Complex ), device );
    cudaMemPrefetchAsync( aqsntemp, number_bands * ncouls * sizeof( Complex ), device );
    cudaMemPrefetchAsync( I_eps_array, ngpown * ncouls * sizeof( Complex ), device );
    cudaMemPrefetchAsync( wtilde_array, ngpown * ncouls * sizeof( Complex ), device );
    cudaMemPrefetchAsync( inv_igp_index, ngpown * sizeof( int ), device );
    cudaMemPrefetchAsync( vcoul, ncouls * sizeof( double ), device );
    cudaMemPrefetchAsync( wx_array, 3 * sizeof( double ), device );
    cudaDeviceSynchronize();

    //Start the timer before the work begins.
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);

    // Complex achstemp(0.0, 0.0);
    // for(int n1 = 0; n1<number_bands; ++n1)
    //{
    //    reduce_achstemp(n1, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp,
    //    ngpown, vcoul);
    //}

    int threads    = 256;
    int block_size = number_bands * ngpown / threads;
    std::cout << "call_compute<<<" << block_size << ", " << threads << ">>>\n"; 
    call_compute<<<block_size, threads>>>( number_bands, ngpown, nvband, inv_igp_index, ncouls,
        wx_array, aqsmtemp, aqsntemp, I_eps_array, wtilde_array, achtemp, asxtemp, vcoul );
    cudaDeviceSynchronize();

    // Time Taken
    gettimeofday(&endTimer, NULL);
    double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);

    // Print results
    for ( int iw = nstart; iw < nend; ++iw )
        cout << "achtemp[" << iw << "] = " << std::setprecision( 15 ) << achtemp[iw] << endl;

    cout << "********** Time Taken **********= " << elapsedTimer << " secs" << endl;

    // Free memory
    cudaFree( inv_igp_index );
    cudaFree( vcoul );
    cudaFree( aqsmtemp );
    cudaFree( aqsntemp );
    cudaFree( I_eps_array );
    cudaFree( wtilde_array );
    cudaFree( achtemp );
    cudaFree( asxtemp );


    return 0;
}

// Almost done code
