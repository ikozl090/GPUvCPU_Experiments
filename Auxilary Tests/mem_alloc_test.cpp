#include <stdio.h> 
#include <stdlib.h> 

#define GB 1073741824
#define MB 1048576

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
    bool* pointer = (bool *)malloc(size);
    printf("Size of one bool is %d bytes\n", sizeof(bool));

    // Check for errors 
    if (pointer == 0) {
        printf("malloc failed!\n");
        return 1; 
    }

    printf("malloc completed succesfully with pointer = %p\n", pointer); // After successful allocation 

    int i; 
    try{
        for (i = 0; i < size; i++) {
            pointer[i] = true;
            if (((i % GB) == 0) && (i > 0)){
                printf("Assigned %d GB\n", ((i + 1) / GB));
            }
        }
    } catch (...) {
        printf("Failed assigning memory at index i = %d\n", i); 
    }

    printf("Successfully filled all allocated memory!\n");

}