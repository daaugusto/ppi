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
void acc_interpret_init( char ** argv, int argc, const unsigned size, const unsigned population_size, float** input, float** model, float* obs, int nlin, int ninput, int nmodel, int mode, const char* type );

/** ************************************************************************************************** **/
/** ************************************** Function interpret **************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void acc_interpret( Symbol* phenotype, float* ephemeral, int* size, float* vector, int nInd, int mode );

#endif
