#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include <stdio.h> 
#include <stdlib.h> 
 
  global   void transpose(int *a, int *t) 
{ 
int n=threadIdx.x, m=blockIdx.x, size=blockDim.x, size1=gridDim.x; 
t[n*size1+m]=a[m*size+n]; 
} 
int main(void) 
{ 
int *a,*t, m,n,i,j; 
int *d_a,*d_t; 
printf("Enter the value of m: ");scanf("%d",&m); 
printf("Enter the value of n: ");scanf("%d",&n); 
int size=sizeof(int)*m*n; 
a=(int*)malloc(m*n*sizeof(int)); 
c=(int*)malloc(m*n*sizeof(int)); 
printf("Enter input matrix:\n"); 
for(i=0;i<m*n;i++) 
scanf("%d",&a[i]); 
 
cudaMalloc((void**)&d_a,size); 
cudaMalloc((void**)&d_t,size); 
cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice); 
transpose<<<m,n>>>(d_a,d_t); 
cudaMemcpy(t,d_t,size,cudaMemcpyDeviceToHost); 
printf("Result vector is:\n"); 
for(i=0;i<n;i++) 
{ 
for(j=0;j<m;j++) 
printf("%d\t",t[i*m+j]); 
printf("\n"); 
} 
getchar(); 
cudaFree(d_a); 
cudaFree(d_t); 
return 0; 
} 