/** TAD: INTERPRET **/

#ifndef accelerator_h
#define accelerator_h

#include <definitions.h>
#include <symbol>
#include "../individual"

/** Funcoes exportadas **/
/** ************************************************************************************************** **/
/** *********************************** Function interpret_init ************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
int acc_interpret_init( int argc, char** argv, const unsigned size, const unsigned max_arity, const unsigned population_size, float** input, int nlin, int ncol, int pep_mode, int prediction_mode );

/** ************************************************************************************************** **/
/** ************************************** Function interpret **************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void acc_interpret( Symbol* phenotype, float* ephemeral, int* size, int sum_size_gen, float* vector, int nInd, void (*send)(Population*), int (*receive)(GENOME_TYPE*), Population* migrants, int* nImmigrants, int* index, int* best_size, int pep_mode, int prediction_mode, float alpha );

/** ************************************************************************************************** **/
/** ************************************** Function print_time *************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void acc_print_time( bool total, int sum_size );

#endif
