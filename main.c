#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "genann.h"

const char *diabities_data = "diabities.csv";

double *input, *class;
int samples;
const char *class_names[] = {"D", "H"};

void load_data() {
    FILE *in = fopen("diabities.csv", "r");
    if (!in) {
        printf("Could not open file: %s\n", diabities_data);
        exit(1);
    }

    // Loop through the data to get a count.
    char line[1024];
    while (!feof(in) && fgets(line, 1024, in)) {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);

    printf("Loading %d data points from %s\n", samples, diabities_data);

    input = (double *) malloc(sizeof(double) * samples * 8);
    class = (double *) malloc(sizeof(double) * samples * 2);

    // Read the file into the arrays.
    int i, j;
    for (i = 0; i < samples; ++i) {
        double *p = input + i * 8;
        double *c = class + i * 2;
        c[0] = c[1] = 0.0;

        if (fgets(line, 1024, in) == NULL) {
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");
        for (j = 0; j < 8; ++j) {
            p[j] = atof(split);
            split = strtok(0, ",");
        }

        split[strlen(split)-1] = 0;
        if (strcmp(split, class_names[0]) == 0) {c[0] = 1.0;}
        else if (strcmp(split, class_names[1]) == 0) {c[1] = 1.0;}
        else {
            printf("Unknown class %s.\n", split);
            exit(1);
        }

        printf("Data point %d is %f %f %f %f %f %f %f %f ->   %f %f\n", i, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], c[0], c[1]);
    }

    fclose(in);
}


int main(int argc, char *argv[])
{
    printf("Train an ANN on Pima Indianas onset diabets dataset using backpropagation.\n");

    //implement writing weights after training to another file
    //also give option to initialize model with some wights
    //FILE *in = fopen("weights.csv", "w");

    srand(time(0));

    load_data();

    genann *ann = genann_init(8, 1, 100, 2);

    int i, j;
    int loops = 4000;
    char ans;
    double user_inp[8];
    double *userinp = user_inp;
    double *userout;

    // Train the network with backpropagation.
    printf("Training for %d loops over data.\n", loops);
    for (i = 0; i < loops; ++i) {
        for (j = 0; j < samples; ++j) {
            genann_train(ann, input + j*8, class + j*2, .001);
        }

    }

    int correct = 0;
    for (j = 0; j < samples; ++j) {
        const double *guess = genann_run(ann, input + j*8);
        if (class[j*2+0] == 1.0) {if (guess[0] > guess[1]) ++correct;}
        else if (class[j*2+1] == 1.0) {if (guess[1] > guess[0]) ++correct;}
        else {printf("Logic error.\n"); exit(1);}
    }

    printf("%d/%d correct (%0.1f%%).\n", correct, samples, (double)correct / samples * 100.0);

    printf("Do you want to give inputs? ");
    scanf("%c",&ans);
    if (ans == 'y') {
        printf("Provide values for Pregnancies, glucose, diastolic, Triceps, Insulin, bmi, dpf, age\n");
        scanf("%lf %lf %lf %lf %lf %lf %lf %lf",&user_inp[0],&user_inp[1],&user_inp[2],&user_inp[3],&user_inp[4],&user_inp[5],&user_inp[6],&user_inp[7]);
        double *userout = genann_run(ann,userinp);
        if (*userout > *(userout + 1)) printf("The person is Diabetic\n");
        else printf("The person is healthy\n");
        //printf("%lf and %lf",userout[0],userout[1]);
    }
    printf("Thank you");

    genann_free(ann);
    free(input);
    free(class);

    return 0;
}
