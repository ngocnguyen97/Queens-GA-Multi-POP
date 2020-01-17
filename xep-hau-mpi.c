#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define POPSIZE 1000
#define CROSSRATE 1.0
#define MUTARATE 0.5
#define MAX_GENERATION 1000
#define MAX_UNCHANGE_GENERATION 200
#define NUM_RUN 10

/*
===========================================================================
Program: solving N-Queens problem using parallel Genetic Algorithm

IDEA:			
	Individual: [coordinate of queens, fitness].
	Eg: N = 8, indi = [0 6 1 3 4 2 5 7 fitness] where fitness is the number of errors
	the smaller fitness, the better result
	the solution should have fitness = 0
===========================================================================
*/

//List function in this Program:
//=========================================================================
//Random init an Indi with size N (queens), return the indi:
int* randomInit(int N);				
//Calculate fitness of indi and add it to the last of indi, return fitness:
int calcFitness(int* indi, int N);	
//Crossover operator, return the child: (part = 0: keep first half, part = 1, keep second half)
int* crossover(int* indi1, int* indi2, int part, int N);
//Mutation, return the mutation indi:	
int* mutation(int* indi, int N);	
//Sort listToSort in each proc, and merge all to proc 0 (size is the local size of listToSort):				
int** sortAndMerge(int** listToSort, int* size, int myid, int numprocs, int N);
//Merge list1 with list2, free them and return the merged list:		
int** merge(int** list1, int size1, int** list2, int size2, int N);
//Also merge list1 with list2, but only select the best size1 indi:	
int** selectionMerge(int** list1, int size1, int** list2, int size2, int N);
//Do quicksort locally:
void quicksort(int** listIndi, int first, int last, int N);	
//Free list and very item of it	
void freeList(int** list, int size);
//Make a clone of indi	
int* cloneIndi(int* indi, int N);
//Print the indi to screen		
void printIndi(int* indi, int N);		
//===========================================================================
int solve(int argc, char *argv[], int N, int myid, int numprocs);

//MPI debug parameters
enum Tag {INDIVIDUAL, LISTSIZE, COMMTIME} messTag;
MPI_Status status;
int err;
double commTime = 0.0;
double wtime = 0.0;

int main(int argc, char *argv[])
{
	//Number of QUEENS
	int Nq;

	int i;
	int numprocs, myid;
	double wt[NUM_RUN], ct[NUM_RUN];
	double sumw = 0.0, sumc = 0.0;

	err = MPI_Init(&argc, &argv);
	if(err != MPI_SUCCESS){
		printf("MPI Init failed!\n");
		exit(1);
	}

	err = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	if(err != MPI_SUCCESS){
		printf("ERR: MPI get id failed!\n");
	}

	err = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	if(err != MPI_SUCCESS){
		printf("ERR: ID %d MPI get comm size failed!\n", myid);
	}
	if(numprocs < 2){
		printf("ERR: This program need at least 2 proccess to run\n");
		return 1;
	}

	if(myid == 0)
	{
		printf("Input the number of Queens: ");
		fflush(stdout);
		scanf("%d", &Nq);
		printf("\n\n=====================================\nStart Parallel Genetic Algorithm\n=====================================\n");
	}

	err = MPI_Bcast(&Nq, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(err != MPI_SUCCESS){
		printf("ERR: broadcast failed !\n");
	}

	for(i=0;i<NUM_RUN;i++){		
		solve(argc, argv, Nq, myid, numprocs);
		wt[i] = wtime;
		ct[i] = commTime;
		sumw += wtime;
		sumc += commTime;
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	if(myid == 0){
		printf("\n==================================FINALIZE=============================\n");
		printf("\t Lan chay \t wtime \t\t commTime \n");
		for(i=0;i<NUM_RUN;i++){
			printf("\t %d \t\t %.4f \t %.4f \n", i, wt[i], ct[i]);
		}
		printf("\nTrung binh \t wtime: %f \t commTime: %f \n",sumw/NUM_RUN, sumc / NUM_RUN );
		FILE *f = fopen("ketqua","at");
		fprintf(f, "\nNumcore: %d \t wtime: %f \t commTime: %f \t commTimeAverage: %f \n",numprocs, sumw/NUM_RUN, sumc / NUM_RUN, sumc / NUM_RUN / numprocs );
		fclose(f);
	}

	//Terminate MPI
	err = MPI_Finalize();
	exit(0);
}

int solve(int argc, char *argv[], int N, int myid, int numprocs)
{
	//Parameter of GA
	int** indiList;
	int** childList;
	int crossNum;
	int mutaNum;

	//MPI parameter
	commTime = 0.0;
	wtime = 0.0;

	//counter:
	int i,j;

	//setup seed for random
	srand(time(NULL) + myid); //random seed

	//LOG the TIME
	if ( myid == 0 )
    {
    	wtime = MPI_Wtime();
    }

	//INIT POPULATION:
	//malloc the memory for population
	indiList = (int**)malloc((POPSIZE)  * sizeof(int*));

	for(i = 0; i<POPSIZE;i++){
		indiList[i] = (int*)malloc((N+1) * sizeof(int));
	}

	enum Tag messTag = INDIVIDUAL;

	//init on id 0
	if(myid == 0){
		for(i = 0; i<POPSIZE;i++){
			indiList[i] = (int*)malloc((N+1) * sizeof(int));
			commTime -= MPI_Wtime();
			err = MPI_Recv(indiList[i], N + 1, MPI_INT, MPI_ANY_SOURCE, messTag, MPI_COMM_WORLD, &status);
			commTime += MPI_Wtime();
			if(err != MPI_SUCCESS)
				printf("\n ERR: ID %d RECIEVE FAILED (SOURCE ID %d)", myid, status.MPI_SOURCE);
		}
		quicksort(indiList, 0, POPSIZE - 1, N);
	}
	else{
		for(i = myid; i<=POPSIZE; i+= numprocs - 1){
			int* temp = randomInit(N);
			commTime -= MPI_Wtime();
			err = MPI_Send(temp, N+1, MPI_INT, 0, messTag, MPI_COMM_WORLD);
			commTime += MPI_Wtime();
			if(err != MPI_SUCCESS)
				printf("\n ERR: ID %d SEND FAILED ", myid);
		}
	}
	//broadcast to everyone
	for(i = 0; i<POPSIZE;i++){
		commTime -= MPI_Wtime();
		err = MPI_Bcast(indiList[i], N+1, MPI_INT, 0, MPI_COMM_WORLD);
		commTime += MPI_Wtime();
		if(err != MPI_SUCCESS)
			printf("\n ERR: broadcast failed");
	}

	
	//MAIN LOOOOOOP:
	int *temp1, *temp2;
	int iter;

	int lastBEST = 10000000;
	int unchangeGen = 0;

	//keep child list separate in each process
	crossNum = (int)(POPSIZE * CROSSRATE / numprocs); 
	mutaNum = (int)(crossNum * MUTARATE);

	for(iter = 0;iter<MAX_GENERATION;iter++){

		//init childList
		int localSize = mutaNum + crossNum;
		childList = (int**)malloc((localSize)  * sizeof(int*));
		for(i = 0; i<localSize;i++){
			childList[i] = (int*)malloc((N+1) * sizeof(int));
		}

		//CROSSOVER OPERATOR: half crossover with part = 0, half with part = 1
		for(j = 0; j < crossNum / 2; j++){ 
			temp1 = indiList[rand() % POPSIZE];
			do
				temp2 = indiList[rand() % POPSIZE];
			while(temp2 == temp1);

			childList[j] = crossover(temp1, temp2, 0, N);
		}
		for(; j < crossNum; j++){ 
			temp1 = indiList[rand() % POPSIZE];
			do
				temp2 = indiList[rand() % POPSIZE];
			while(temp2 == temp1);

			childList[j] = crossover(temp1, temp2, 1, N);
		}

		//MUTATION OPERATOR:
		for(j = crossNum; j < localSize; j++){ 
			temp1 = childList[rand() % crossNum];
			childList[j] = mutation(temp1, N);
		}

		//SELECTION:
		//merge all child list of each proc to proc 0 (and free all other list)
		childList = sortAndMerge(childList, &localSize, myid, numprocs, N);

		//merge with pop in id 0, and then select best POPSIZE indi
		if(myid == 0){
			indiList = selectionMerge(indiList, POPSIZE, childList, localSize, N);
		}

		//broadcast to everyone
		for(i = 0; i<POPSIZE;i++){
			commTime -= MPI_Wtime();
			err = MPI_Bcast(indiList[i], N+1, MPI_INT, 0, MPI_COMM_WORLD);
			commTime += MPI_Wtime();
			if(err != MPI_SUCCESS)
				printf("\n ERR: broadcast failed");
		}
		//finish selection

		//print debug line
		if(myid == 0){
			printf("Iteration : %d : best finess: %d \n",iter, indiList[0][N]);	
		}

		//CHECK STOP CONDITION
		if(indiList[0][N] == lastBEST){
			unchangeGen++;
		}
		else{
			unchangeGen = 0;
			lastBEST = indiList[0][N];
		}
		if(indiList[0][N] == 0 || unchangeGen == MAX_UNCHANGE_GENERATION)
			break;

		//END LOOP i
	}
	

	//print solution
	if(myid == 0){
		wtime = MPI_Wtime ( ) - wtime;
		printf("\n=================================================  ");
		printf("\n-> Solution is: \n ");
		printIndi(indiList[0], N);
		printf("-> Time to solve is: %10f second \n", wtime);
		printf("-> Communication Time of each procs in second: \n");
		printf("\tId %d : %10f \n", myid, commTime);
		messTag = COMMTIME;
		for(i =1;i<numprocs;i++){
			double ct;
			err = MPI_Recv(&ct,1 , MPI_DOUBLE, MPI_ANY_SOURCE, messTag, MPI_COMM_WORLD, &status);
			if(err != MPI_SUCCESS)
				printf("\n ERR: ID %d RECEIVE FAILED (SOURCE ID %d)", myid, status.MPI_SOURCE);
			printf("\tId %d : %10f \n", status.MPI_SOURCE, ct);
			commTime+= ct;
		}
		printf("-> Total Communication time is : %10f second \n", commTime);
	}
	else{
		messTag = COMMTIME;
		err = MPI_Send(&commTime, 1, MPI_DOUBLE, 0, messTag, MPI_COMM_WORLD);
		if(err != MPI_SUCCESS)
			printf("\n ERR: ID %d SEND FAILED ", myid);
	}

	freeList(indiList, POPSIZE);

	return 0;
}



int* randomInit(int N)
{	
	int i,j;
	int* indi = (int*)malloc((N+1) * sizeof(int));
    // initial range of numbers
    for(i=0;i<N;i++){
        indi[i]=i;
    }
    // shuffle
    for (i = 0; i < N; i++){
        j = rand() % N;
        int temp = indi[i];
        indi[i] = indi[j];
        indi[j] = temp;
  	}

  	indi[N] = calcFitness(indi, N);

  	return indi;
} 

int calcFitness(int* indi, int N)
{
	int fit = 0;
	int i,j;

	for(i = 0; i<N;i++){
		for(j=i+1;j<N;j++){
			if(abs(indi[j] - indi[i]) == j - i)
				fit++;
		}
	}

	return fit;
}

int* crossover(int* indi1, int* indi2, int part, int N)
{
	int i,j;
	int* child = (int*)malloc((N+1) * sizeof(int));
	int* temp = (int*)malloc((N+1) * sizeof(int));
	int index;
	
	for(i=0;i <= N;i++){
		temp[i] = indi2[i];
	}
	if(part == 0){
		index = 1 + rand() % (N - 3);
		for(i=0;i <= index;i++){
			child[i] = indi1[i];
			for(j=0;j<N;j++){
				if(temp[j] == child[i]){
					temp[j] = -1; 
					break;
				}
			}
		}
		for(j=0;j<N;j++){
			if(temp[j] != -1){
				child[i] = temp[j];
				i++; 
				if(i == N) 
					break;
			}
		}
	}
	else{
		index = 2 + rand() % (N - 3);
		for(i=index;i < N;i++){
			child[i] = indi1[i];
			for(j=0;j<N;j++){
				if(temp[j] == child[i]){
					temp[j] = -1; 
					break;
				}
			}
		}
		i = 0;
		for(j=0;j<N;j++){
			if(temp[j] != -1){
				child[i] = temp[j];
				i++; 
				if(i == index) 
					break;
			}
		}
	}
	free(temp);

	child[N] = calcFitness(child, N);
	
	return child;
}

int* mutation(int* indi, int N)
{
	int* child = (int*)malloc((N+1) * sizeof(int));
	int index1 = rand() % (N);
	int index2 = rand() % (N);
	int i, temp;

	for(i=0;i<= N;i++){
		child[i] = indi[i];
	}

	temp = child[index1];
	child[index1] = child[index2];
	child[index2] = temp;

	child[N] = calcFitness(child, N);

	return child;
}

int** sortAndMerge(int** listToSort, int* size, int myid, int numprocs, int N)
{
	int i,j;
	//sort locally
	quicksort(listToSort, 0, *size - 1, N);
	//merge
	int* aProc = (int*)malloc(numprocs*sizeof(int));
	for(i =0;i<numprocs;i++)
		aProc[i] = i;
	int aNumProc = numprocs;
	int localSize = *size;
	int** temp;
	int sizeTemp;

	while(aNumProc != 1){	
		int nextaNum;
		if(aNumProc % 2 == 0)
			nextaNum = aNumProc / 2;
		else 
			nextaNum = aNumProc / 2 + 1;
		int* nextaProc = (int*)malloc(nextaNum*sizeof(int));
			
		for(i = 0; i<aNumProc;i+= 2){
			nextaProc[i/2] = aProc[i];
			if(i == aNumProc - 1){
				continue;
			}

			int proc1 = aProc[i];
			int proc2 = aProc[i+1];
			if(myid == proc1){
				messTag = LISTSIZE;
				commTime -= MPI_Wtime();
				err = MPI_Recv(&sizeTemp, 1, MPI_INT, proc2, messTag, MPI_COMM_WORLD, &status);
				commTime += MPI_Wtime();
				if(err != MPI_SUCCESS)
					printf("\n ERR: ID %d RECIEVE FAILED (SOURCE ID %d)", myid, status.MPI_SOURCE);
				temp = (int**)malloc(sizeTemp*sizeof(int*));
					messTag = INDIVIDUAL;
				for(j =0; j<sizeTemp;j++){
					temp[j] = (int*)malloc((N+1) * sizeof(int));
					err = MPI_Recv(temp[j], N+1, MPI_INT, proc2, messTag, MPI_COMM_WORLD, &status);
					if(err != MPI_SUCCESS)
						printf("\n ERR: ID %d RECIEVE FAILED (SOURCE ID %d)", myid, status.MPI_SOURCE);
				}
				
				listToSort = merge(listToSort, localSize, temp, sizeTemp, N);
				localSize = localSize + sizeTemp;
			}
			if(myid == proc2){
				messTag = LISTSIZE;
				commTime -= MPI_Wtime();
				err = MPI_Send(&localSize, 1, MPI_INT, proc1, messTag, MPI_COMM_WORLD);
				commTime += MPI_Wtime();
				if(err != MPI_SUCCESS)
					printf("\n ERR: ID %d SEND FAILED ", myid);
					messTag = INDIVIDUAL;
				for(j =0; j<localSize;j++){
					err = MPI_Send(listToSort[j], N+1, MPI_INT, proc1, messTag, MPI_COMM_WORLD);
					if(err != MPI_SUCCESS)
						printf("\n ERR: ID %d SEND FAILED ", myid);
				}
				freeList(listToSort, localSize);
			}
		}
		aProc = nextaProc;
		aNumProc = nextaNum;
	}

	*size = localSize;
	free(aProc);

	return listToSort;
	
}

int** merge(int** list1, int size1, int** list2, int size2, int N)
{
	int** mList = (int**)malloc((size1+size2) * sizeof(int*));

	int count = 0;
	int i = 0;
	int j = 0;

	while(count < size1 + size2){
		if(i==size1){
			mList[count] = cloneIndi(list2[j], N);
			j++;
			count++;
			continue;
		}
		if(j==size2){
			mList[count] = cloneIndi(list1[i], N);
			i++;
			count++;
			continue;
		}
		if(list1[i][N] <= list2[j][N]){
			mList[count] = cloneIndi(list1[i], N);
			i++;
			count++;
			continue;
		}
		else {
			mList[count] = cloneIndi(list2[j], N);
			j++;
			count++;
			continue;
		}
	}
	freeList(list1, size1);
	freeList(list2, size2);

	return mList;
}

int** selectionMerge(int** list1, int size1, int** list2, int size2, int N)
{
	int** mList = (int**)malloc((size1) * sizeof(int*));
	int count = 0;
	int i = 0;
	int j = 0;

	while(count < size1){
		if(i==size1){
			mList[count] = cloneIndi(list2[j], N);
			j++;
			count++;
			continue;
		}
		if(j==size2){
			mList[count] = cloneIndi(list1[i], N);
			i++;
			count++;
			continue;
		}
		if(list1[i][N] <= list2[j][N]){
			mList[count] = cloneIndi(list1[i], N);
			i++;
			count++;
			continue;
		}
		else {
			mList[count] = cloneIndi(list2[j], N);
			j++;
			count++;
			continue;
		}
	}
	freeList(list1, size1);
	freeList(list2, size2);

	return mList;
}

void quicksort(int** listIndi, int first, int last, int N)
{
    int i, j, pivot, *temp;

    if(first < last){
      	pivot = first;
      	i = first;
      	j = last;

      	while(i < j){
         	while(listIndi[i][N] <= listIndi[pivot][N] && i<last)
            	i++;
          	while(listIndi[j][N] > listIndi[pivot][N])
            	j--;
        	if(i < j){
            	temp = listIndi[i];
            	listIndi[i] = listIndi[j];
            	listIndi[j] = temp;
        	}
      	}

      	temp = listIndi[pivot] ;
      	listIndi[pivot] = listIndi[j];
      	listIndi[j] = temp;

      	quicksort(listIndi,first,j-1,N);
      	quicksort(listIndi,j+1,last,N);
    }
}

int* cloneIndi(int* indi, int N)
{
	int* child = (int*)malloc((N+1) * sizeof(int));
	int i;

	for(i=0;i< N+1;i++){
		child[i] = indi[i];
	}

	return child;
}

void freeList(int** list, int size)
{
	int i;
	for(i=0;i<size;i++)
		free(list[i]);
	free(list);
}

void printIndi(int* indi, int N)
{
	int i;

	for(i=0;i<N;i++){
		printf("%d ", indi[i] + 1);
	}
	printf(" || fitness: %d \n", indi[N]);

	return;
}