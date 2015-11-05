#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include "utils.h"

void go_to_work( double **localA, double *k_row, int l_size, int y, int rank, int owner, int start, int k ){
    int i, j;
    double l;
    
    /* osoi exoun rank mikrotero tou owner exoun teleiwsei th douleia tous */
    if (rank < owner)
	return;
    
    /* oi alloi kanoun tous ypologismous tous, opws sto serial */
    for (i=start; i < l_size; i++){
	l = localA[i][k] / k_row[k];
	for (j=k; j<y; j++)
	    localA[i][j] -= l*k_row[j];
    }
}

int main (int argc, char * argv[]) {
    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int X,Y,x,y,X_ext;
    double ** A, ** localA;
    X=atoi(argv[1]);
    Y=X;

    //Extend dimension X with ghost cells if X%size!=0
    if (X%size!=0)
        X_ext=X+size-X%size;
    else
        X_ext=X;
     if (rank==0) {
        //Allocate and init matrix A
        A=malloc2D(X_ext,Y);
        init2D(A,X,Y);
    }

      
    //Local dimensions x,y
    x=X_ext/size;
    y=Y;

    //Allocate local matrix and scatter global matrix
    localA=malloc2D(x,y);
    double * idx;
    if (rank==0)
        idx=&A[0][0];
    MPI_Scatter(idx,x*y,MPI_DOUBLE,&localA[0][0],x*y,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if (rank==0)
        free2D(A,X_ext,Y);
 
    //Timers   
    struct timeval ts, tf, comps, compf, comms, commf;
    double total_time, computation_time=0, communication_time=0;

    
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&ts,NULL);        
    /******************************************************************************
     The matrix A is distributed in contiguous blocks to the local matrices localA
     You have to use collective communication routines.
     Don't forget the timers for computation and communication!
        
    ******************************************************************************/
    
    /* kathe enas krataei se enan pinaka y thesewn (temp) thn k_row pou kanei
     * broadcast o owner */
    int line_index, line_owner;
    int k, start;
    double *k_row, *temp;
    
    temp = malloc(y * sizeof(*temp));
//     k_row = malloc(y * sizeof(*k_row));
    
    for (k=0; k<y-1; k++){
	start = 0;
	line_owner = k / x;
	line_index = k % x;
	
	/* o line_owner prepei na efarmosei tous ypologismous tou sto localA
	 * kai katw apo th grammh k */
	if (rank == line_owner){
	    start = line_index+1;
	    k_row = localA[line_index];
	}
	else
	    k_row = temp;
	
	/* set communication timer */
	gettimeofday(&comms, NULL);
	
	/* k_line_owner broadcasts */
	MPI_Bcast(k_row, y, MPI_DOUBLE, line_owner, MPI_COMM_WORLD);
	
	/* stop communication timer */
	gettimeofday(&commf, NULL);
	communication_time += commf.tv_sec - comms.tv_sec + (commf.tv_usec - comms.tv_usec)*0.000001;
	
	/* set computation timer */
	gettimeofday(&comps, NULL);
	
	/* Compute */
	go_to_work( localA, k_row, x, y, rank, line_owner, start, k );
	
	/* stop computation timer */
	gettimeofday(&compf, NULL);
	computation_time += compf.tv_sec - comps.tv_sec + (compf.tv_usec - comps.tv_usec)*0.000001;
    }
    
    gettimeofday(&tf,NULL);
    total_time=tf.tv_sec-ts.tv_sec+(tf.tv_usec-ts.tv_usec)*0.000001;


    //Gather local matrices back to the global matrix
    if (rank==0) {
        A=malloc2D(X_ext,Y);
        idx=&A[0][0];
    }
    MPI_Gather(&localA[0][0],x*y,MPI_DOUBLE,idx,x*y,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    double avg_total,avg_comp,avg_comm,max_total,max_comp,max_comm;
    MPI_Reduce(&total_time,&max_total,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&computation_time,&max_comp,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&communication_time,&max_comm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&total_time,&avg_total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&computation_time,&avg_comp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&communication_time,&avg_comm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    avg_total/=size;
    avg_comp/=size;
    avg_comm/=size;

    if (rank==0) {
        printf("LU-Block-bcast\tSize\t%d\tProcesses\t%d\n",X,size);
        printf("Max times:\tTotal\t%lf\tComp\t%lf\tComm\t%lf\n",max_total,max_comp,max_comm);
        printf("Avg times:\tTotal\t%lf\tComp\t%lf\tComm\t%lf\n",avg_total,avg_comp,avg_comm);
    }

    //Print triangular matrix U to file
    int print = 1;
    if (print)
	if (rank==0) {
	    char * filename="output_block_bcast";
	    print2DFile(A,X,Y,filename);	// print2DFile(A, X, Y, filename);qy
	     
	}
    
    MPI_Finalize();

    return 0;
}


