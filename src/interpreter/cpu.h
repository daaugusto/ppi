/** TAD: INTERPRET **/

#ifndef cpu_h
#define cpu_h

#include "../symbol"

/** Funcoes exportadas **/
/** ************************************************************************************************** **/
/** *********************************** Function interpret_init ************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void interpret_init( const unsigned max_size_phenotype, double** input, double** model, double* obs, int nlin, int ninput, int nmodel );

/** ************************************************************************************************** **/
/** ************************************** Function interpret **************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
double interpret( Symbol* phenotype, double* ephemeral, int size );

#endif
