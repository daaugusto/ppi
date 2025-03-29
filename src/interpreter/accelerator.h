////////////////////////////////////////////////////////////////////////////////
//  Parallel Program Induction (PPI) is free software: you can redistribute it
//  and/or modify it under the terms of the GNU General Public License as
//  published by the Free Software Foundation, either version 3 of the License,
//  or (at your option) any later version.
//
//  PPI is distributed in the hope that it will be useful, but WITHOUT ANY
//  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
//  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
//  details.
//
//  You should have received a copy of the GNU General Public License along
//  with PPI.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////////////////

#ifndef accelerator_h
#define accelerator_h

#include <definitions.h>
#include <symbol>
#include "../individual"

namespace ppi {

/** Funcoes exportadas **/
/** ************************************************************************************************** **/
/** *********************************** Function interpret_init ************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
int acc_interpret_init( int argc, char** argv, const unsigned size, const unsigned max_arity, const unsigned population_size, float** input, int nlin, int ncol, int ppp_mode, int prediction_mode );

/** ************************************************************************************************** **/
/** ************************************** Function interpret **************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
void acc_interpret( Symbol* phenotype, float* ephemeral, int* size, 
#ifdef PROFILING
unsigned long sum_size_gen, 
#endif
float* vector, int nInd, void (*send)(Population*), int (*receive)(GENOME_TYPE**), Population* migrants, int* nImmigrants, int* index, int* best_size, int ppp_mode, int prediction_mode, float alpha );

/** ************************************************************************************************** **/
/** ************************************** Function print_time *************************************** **/
/** ************************************************************************************************** **/
/**                                                                                                    **/
/** ************************************************************************************************** **/
#ifdef PROFILING
void acc_print_time( bool total, unsigned long long sum_size );
#endif

}

#endif
