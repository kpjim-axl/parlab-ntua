#include <stdlib.h>
#include "utils.h"

int main(int argc, char **argv){
    int x = atoi(argv[1]);
    int y = x;
    
    double ** A = malloc2D(x, y);
    init2D(A, x, y);
    
    print2DFile(A, x, y, argv[2]);
    free2D(A, x, y);
    return 0;
}