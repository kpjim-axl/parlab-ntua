/**********************************************************************
    Modify the code-Add OpenMP directives to parallelize the LU kernel
***********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "utils.h"

int main(int argc, char * argv[]){
    int i,j,k;
    double l;
    double total_time;
    struct timeval ts,tf;

    int X = atoi(argv[1]);
    int Y = X;
    double **A=malloc2D(X,Y);
    double *Ak, *Ai;
//     print2D(A, X, Y);
    init2D(A, X, Y);

    gettimeofday(&ts,NULL);
    
    /* Theloume oso ginetai, na kanoume liges anafores sthn koinh mnhmh..
     * Opote kratame tis ekastote grammes k kai i me 3exwristous private 
     * deiktes Ak kai Ai antistoixa */
    for (k=0; k<X-1; k++)
    {									// for k
#pragma omp parallel private(Ak)
{									// pragma
	Ak = A[k];
#pragma omp for schedule(static) private(l, j, Ai)
    for (i=k+1; i<X; i++) {						// for i
	Ai = A[i];
	l=Ai[k]/Ak[k];
	for (j=k; j<Y; j++)
	    Ai[j]-=l*Ak[j];
    }									// for i
}									// pragma
    }									// for k
    
    gettimeofday(&tf,NULL);
    total_time=(tf.tv_sec-ts.tv_sec)+(tf.tv_usec-ts.tv_usec)*0.000001;
    
    printf("LU-OpenMP\t%d\t%.3lf\n",X,total_time);
    
    char * filename="output_omp";
    print2DFile(A,X,Y,filename);
    free2D(A, X, Y);
    return 0;
}
