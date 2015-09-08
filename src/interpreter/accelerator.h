/** TAD: INTERPRET **/

#ifndef accelerator_h
#define accelerator_h

#include "../symbol"

/** Funcoes exportadas **/
/** ************************************************************************************************** **/
/** *********************************** Function interpret_init ************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
int acc_interpret_init( int argc, char** argv, const unsigned size, const unsigned population_size, float** input, float** model, float* obs, int nlin, int prediction_mode );

/** ************************************************************************************************** **/
/** ************************************** Function interpret **************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void acc_interpret( Symbol* phenotype, float* ephemeral, int* size, float* vector, int nInd, int (*function)(int*), int* immigrants, int* nImmigrants, int* index, int* best_size, int prediction_mode, int alpha );

/** ************************************************************************************************** **/
/** ************************************** Function print_time *************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void acc_print_time();

#endif
