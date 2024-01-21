#include <stdio.h> 
#include <stdlib.h> 

#define GB 1073741824

int main(int argc, char *argv[]) {

    // Ensure matrix dimension was given
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_number> (Please enter size in GB to allocated)\n", argv[0]);
        return 1;
    }

    long long int size = atoi(argv[1]);  // Convert the argument to an integer
    size = size * GB; // Convert to GB

    printf("Allocating memory (%lld bytes)...\n", size); // Before allocation

    // Allocate memory 
    void* pointer = malloc(size);

    // Check for errors 
    if (pointer == 0) {
        printf("malloc failed!\n");
        return 1; 
    }

    printf("malloc completed succesfully with pointer = %p\n", pointer); // After successful allocation 

}