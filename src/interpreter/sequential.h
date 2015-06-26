/** TAD: INTERPRET **/

#ifndef sequential_h
#define sequential_h

#include "../symbol"

/** Funcoes exportadas **/
/** ************************************************************************************************** **/
/** *********************************** Function interpret_init ************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void seq_interpret_init( const unsigned size, float** input, float** model, float* obs, int nlin, int ninput, int nmodel );

/** ************************************************************************************************** **/
/** ************************************** Function interpret **************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void seq_interpret( Symbol* phenotype, float* ephemeral, int* size, float* vector, int nInd, int prediction_mode );

/** ************************************************************************************************** **/
/** ********************************** Function interpret_destroy ************************************ **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void seq_interpret_destroy();

#endif
