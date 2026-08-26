#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); exit(1);} } while(0)
#define FFT_OK(x) do { cufftResult e=(x); if(e!=CUFFT_SUCCESS){fprintf(stderr,"cuFFT error: %d\n",e); exit(1);} } while(0)
constexpr float PI = 3.14159265358979323846f;

__device__ int mode(int i, int n) { return i <= n / 2 ? i : i - n; }
__device__ bool kept(int x, int y, int z, int n) {
    return abs(mode(x,n)) <= n/3 && abs(mode(y,n)) <= n/3 && z <= n/3;
}

__global__ void initialize(float *u, int n) {
    int q=blockIdx.x*blockDim.x+threadIdx.x, n3=n*n*n;
    if(q>=n3) return;
    int z=q%n, y=(q/n)%n, x=q/(n*n);
    float X=2.0f*PI*x/n, Y=2.0f*PI*y/n, Z=2.0f*PI*z/n;
    u[q]=sinf(X)*cosf(Y)*cosf(Z);
    u[n3+q]=-cosf(X)*sinf(Y)*cosf(Z);
    u[2*n3+q]=0.0f;
}

__global__ void filter(cufftComplex *a, int n) {
    int q=blockIdx.x*blockDim.x+threadIdx.x, nc=n*n*(n/2+1);
    if(q>=3*nc) return;
    int p=q%nc, z=p%(n/2+1), y=(p/(n/2+1))%n, x=p/(n*(n/2+1));
    if(!kept(x,y,z,n)) a[q]=make_cuFloatComplex(0,0);
}

__global__ void derivatives(cufftComplex *out, const cufftComplex *in, int n) {
    int q=blockIdx.x*blockDim.x+threadIdx.x, nc=n*n*(n/2+1);
    if(q>=3*nc) return;
    int p=q%nc, z=p%(n/2+1), y=(p/(n/2+1))%n, x=p/(n*(n/2+1));
    float kx=mode(x,n), ky=mode(y,n), kz=z;
    cufftComplex v=in[q];
    out[q]=make_cuFloatComplex(-kx*v.y,kx*v.x);
    out[3*nc+q]=make_cuFloatComplex(-ky*v.y,ky*v.x);
    out[6*nc+q]=make_cuFloatComplex(-kz*v.y,kz*v.x);
}

static cufftHandle make_plan_many(int n, int batch, cufftType type) {
    cufftHandle plan;
    int dims[3]={n,n,n};
    int n3=n*n*n, nc=n*n*(n/2+1);
    if(type==CUFFT_R2C) {
        FFT_OK(cufftPlanMany(&plan,3,dims,nullptr,1,n3,nullptr,1,nc,CUFFT_R2C,batch));
    } else {
        FFT_OK(cufftPlanMany(&plan,3,dims,nullptr,1,nc,nullptr,1,n3,CUFFT_C2R,batch));
    }
    return plan;
}

__global__ void nonlinear(const float *u, const float *g, float *nlin, int n3) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=n3) return;
    float ux=u[q], uy=u[n3+q], uz=u[2*n3+q];
    for(int c=0;c<3;c++) {
        const float *gx=g+(3*c+0)*n3, *gy=g+(3*c+1)*n3, *gz=g+(3*c+2)*n3;
        nlin[c*n3+q]=ux*gx[q]+uy*gy[q]+uz*gz[q];
    }
}

__global__ void transpose_gradients(const float *input, float *output, int n3) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=9*n3) return;
    int direction=q/(3*n3), component=(q/n3)%3, point=q%n3;
    output[(3*component+direction)*n3+point]=input[(3*direction+component)*n3+point];
}

__global__ void project_and_diffuse(cufftComplex *out, const cufftComplex *nlin,
                                    const cufftComplex *uh, int n, float nu) {
    int q=blockIdx.x*blockDim.x+threadIdx.x, nc=n*n*(n/2+1);
    if(q>=3*nc) return;
    int p=q%nc, z=p%(n/2+1), y=(p/(n/2+1))%n, x=p/(n*(n/2+1));
    float kx=mode(x,n), ky=mode(y,n), kz=z, k2=kx*kx+ky*ky+kz*kz;
    cufftComplex a=nlin[q], b=nlin[nc+q], c=nlin[2*nc+q];
    float dr=kx*a.x+ky*b.x+kz*c.x, di=kx*a.y+ky*b.y+kz*c.y;
    float k=q<nc?kx:(q<2*nc?ky:kz);
    cufftComplex v=uh[q];
    float pr=k2>0?dr*k/k2:0, pi=k2>0?di*k/k2:0;
    out[q].x=-a.x-pr-nu*k2*v.x;
    out[q].y=-a.y-pi-nu*k2*v.y;
    // Recompute the selected component for the y and z batches.
    if(q>=nc) {
        cufftComplex selected=q<2*nc?b:c;
        out[q].x=-selected.x-(k2>0?dr*k/k2:0)-nu*k2*v.x;
        out[q].y=-selected.y-(k2>0?di*k/k2:0)-nu*k2*v.y;
    }
}

__global__ void scale(float *a, int count, float s) {
    int q=blockIdx.x*blockDim.x+threadIdx.x; if(q<count) a[q]*=s;
}
__global__ void axpy(float *out,const float *base,const float *a,float factor,int count) {
    int q=blockIdx.x*blockDim.x+threadIdx.x; if(q<count) out[q]=base[q]+factor*a[q];
}
__global__ void rk4(float *u,const float *a,const float *b,const float *c,const float *d,float dt,int count) {
    int q=blockIdx.x*blockDim.x+threadIdx.x; if(q<count) u[q]+=dt*(a[q]+2*b[q]+2*c[q]+d[q])/6.0f;
}

struct Solver {
    int n,n3,nc; float dt,nu; float *u,*tmp,*k1,*k2,*k3,*k4,*nlin,*grad,*grad_raw; cufftComplex *uh,*nh,*work,*grad_hat;
    cufftHandle r2c,c2r,c2r9;
    Solver(int N,float step,float viscosity):n(N),n3(N*N*N),nc(N*N*(N/2+1)),dt(step),nu(viscosity) {
        CUDA_OK(cudaMalloc(&u,3ull*n3*sizeof(float))); CUDA_OK(cudaMalloc(&tmp,3ull*n3*sizeof(float)));
        CUDA_OK(cudaMalloc(&k1,3ull*n3*sizeof(float))); CUDA_OK(cudaMalloc(&k2,3ull*n3*sizeof(float)));
        CUDA_OK(cudaMalloc(&k3,3ull*n3*sizeof(float))); CUDA_OK(cudaMalloc(&k4,3ull*n3*sizeof(float)));
        CUDA_OK(cudaMalloc(&nlin,3ull*n3*sizeof(float))); CUDA_OK(cudaMalloc(&grad,9ull*n3*sizeof(float))); CUDA_OK(cudaMalloc(&grad_raw,9ull*n3*sizeof(float)));
        CUDA_OK(cudaMalloc(&uh,3ull*nc*sizeof(cufftComplex))); CUDA_OK(cudaMalloc(&nh,3ull*nc*sizeof(cufftComplex)));
        CUDA_OK(cudaMalloc(&work,3ull*nc*sizeof(cufftComplex))); CUDA_OK(cudaMalloc(&grad_hat,9ull*nc*sizeof(cufftComplex)));
        r2c=make_plan_many(n,3,CUFFT_R2C); c2r=make_plan_many(n,3,CUFFT_C2R); c2r9=make_plan_many(n,9,CUFFT_C2R);
        int b=(n3+255)/256; initialize<<<b,256>>>(u,n); CUDA_OK(cudaGetLastError());
    }
    ~Solver(){ cufftDestroy(r2c); cufftDestroy(c2r); cufftDestroy(c2r9); cudaFree(u);cudaFree(tmp);cudaFree(k1);cudaFree(k2);cudaFree(k3);cudaFree(k4);cudaFree(nlin);cudaFree(grad);cudaFree(grad_raw);cudaFree(uh);cudaFree(nh);cudaFree(work);cudaFree(grad_hat); }
    void transform_r2c(const float *in,cufftComplex *out){ FFT_OK(cufftExecR2C(r2c,const_cast<float*>(in),out)); }
    void transform_c2r(cufftComplex *in,float *out){ FFT_OK(cufftExecC2R(c2r,in,out)); int b=(3*n3+255)/256; scale<<<b,256>>>(out,3*n3,1.0f/n3); }
    void transform_c2r9(cufftComplex *in,float *out){ FFT_OK(cufftExecC2R(c2r9,in,out)); int b=(9*n3+255)/256; scale<<<b,256>>>(out,9*n3,1.0f/n3); }
    void rhs(const float *state,float *out){
        int rb=(n3+255)/256, cb=(3*nc+255)/256;
        transform_r2c(state,uh); filter<<<cb,256>>>(uh,n);
        derivatives<<<cb,256>>>(grad_hat,uh,n); transform_c2r9(grad_hat,grad_raw);
        transpose_gradients<<<(9*n3+255)/256,256>>>(grad_raw,grad, n3);
        nonlinear<<<rb,256>>>(state,grad,nlin); transform_r2c(nlin,nh); filter<<<cb,256>>>(nh,n);
        project_and_diffuse<<<cb,256>>>(work,nh,uh,n,nu); transform_c2r(work,out); CUDA_OK(cudaGetLastError());
    }
    void step(){ int count=3*n3,b=(count+255)/256; rhs(u,k1); axpy<<<b,256>>>(tmp,u,k1,.5f*dt,count); rhs(tmp,k2); axpy<<<b,256>>>(tmp,u,k2,.5f*dt,count); rhs(tmp,k3); axpy<<<b,256>>>(tmp,u,k3,dt,count); rhs(tmp,k4); rk4<<<b,256>>>(u,k1,k2,k3,k4,dt,count); CUDA_OK(cudaGetLastError()); }
    void diagnostics(int step,float t) const {
        std::vector<float> h(3ull*n3); CUDA_OK(cudaMemcpy(h.data(),u,h.size()*sizeof(float),cudaMemcpyDeviceToHost));
        double e=0,zeta=0; for(int q=0;q<n3;q++){e+=.5*(h[q]*h[q]+h[n3+q]*h[n3+q]+h[2*n3+q]*h[2*n3+q]);}
        e/=n3; for(int x=0;x<n;x++) for(int y=0;y<n;y++) for(int z=0;z<n;z++){int q=x*n*n+y*n+z;int xp=((x+1)%n)*n*n+y*n+z,xm=((x+n-1)%n)*n*n+y*n+z;int yp=x*n*n+((y+1)%n)*n+z,ym=x*n*n+((y+n-1)%n)*n+z;int zp=x*n*n+y*n+(z+1)%n,zm=x*n*n+y*n+(z+n-1)%n;float dvydz=(h[n3+zp]-h[n3+zm])/(4*PI/n),dudz=(h[zp]-h[zm])/(4*PI/n),dvdx=(h[n3+xp]-h[n3+xm])/(4*PI/n),dudy=(h[yp]-h[ym])/(4*PI/n);float wx=-dvdx,wy=-dudz,wz=dvydz-dudy;zeta+=.5*(wx*wx+wy*wy+wz*wz);} zeta/=n3;
        printf("Save time step: %d, time: %.6f, enstrophy: %.8e, kinetic energy: %.8e\n",step,t,zeta,e);
    }
};

int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):128;float dt=argc>2?atof(argv[2]):.005f,re=argc>3?atof(argv[3]):1000.f,total=argc>4?atof(argv[4]):1.f,save=argc>5?atof(argv[5]):.05f;if(n<4||dt<=0||re<=0||total<=0||save<=0){fprintf(stderr,"Usage: %s [N] [dt] [Re] [total_time] [save_time]\n",argv[0]);return 1;}Solver s(n,dt,2*PI/re);int steps=(int)(total/dt),every=std::max(1,(int)(save/dt));printf("CUDA Taylor-Green: N=%d Re=%.1f steps=%d\n",n,re,steps);for(int i=1;i<=steps;i++){s.step();if(i%every==0||i==steps)s.diagnostics(i,i*dt);}CUDA_OK(cudaDeviceSynchronize());}
