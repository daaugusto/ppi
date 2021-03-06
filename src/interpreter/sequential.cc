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

#include <stdio.h> 
#include <stdlib.h>
#include <iostream>
#include <cmath>    
#include <string>   
#include <limits>
#include <queue>
#include "sequential.h"
#include "../util/Util.h"

/** ****************************************************************** **/
/** ***************************** TYPES ****************************** **/
/** ****************************************************************** **/

static struct t_data { unsigned size; float** inputs; int nlin; int ncol; double time_total_kernel1; double time_total_kernel2; double time_gen_kernel1; double time_gen_kernel2; double gpops_gen_kernel; } data;

float native_divide(float x, float y) {
   return x / y;
}

float native_sin(float x) {
   return sin(x);
}

float native_cos(float x) {
   return cos(x);
}

float native_tan(float x) {
   return tan(x);
}

float native_sqrt(float x) {
   return sqrt(x);
}

float native_exp(float x) {
   return exp(x);
}

float native_exp10(float x) {
   return exp10(x);
}

float native_exp2(float x) {
   return exp2(x);
}

float native_log(float x) {
   return log(x);
}

float native_log10(float x) {
   return log10(x);
}

float native_log2(float x) {
   return log2(x);
}

#include "functions.h"

/** ****************************************************************** **/
/** ************************* MAIN FUNCTION ************************** **/
/** ****************************************************************** **/

void seq_interpret_init( const unsigned size, float** input, int nlin, int ncol ) 
{
#ifdef PROFILING
   data.time_total_kernel1  = 0.0;
   data.time_total_kernel2  = 0.0;
#endif

   data.size = size;
   data.nlin = nlin;
   data.ncol = ncol;

   data.inputs = new float*[nlin];
   for( int i = 0; i < nlin; i++ )
     data.inputs[i] = new float[ncol];

   for( int i = 0; i < nlin; i++ )
   {
     for( int j = 0; j < ncol; j++ )
     {
       data.inputs[i][j] = input[i][j];
     }
   }

//   for( int i = 0; i < nlin; i++ )
//   {
//      if( i == 289 )
//      {
//         for( int j = 0; j < ncol; j++ )
//         {
//            fprintf(stdout,"%f ",data.inputs[i][j]);
//         }
//      }
//   }
}

void seq_interpret( Symbol* phenotype, float* ephemeral, int* size, 
#ifdef PROFILING
unsigned long sum_size_gen, 
#endif
float* vector, int nInd, int* index, int* best_size, int ppp_mode, int prediction_mode, float alpha )
{
#ifdef PROFILING
   util::Timer t_kernel;
#endif
   // Include the cost matrix definition if given
   #include "costmatrix"

   float stack[data.size]; 
   float sum; 
   int stack_top;

   for( int ind = 0; ind < nInd; ++ind )
   {
      if( size[ind] == 0 && !prediction_mode )
      {
         vector[ind] = std::numeric_limits<float>::max();
         continue;
      }

      sum = 0.0;
      for( int ponto = 0; ponto < data.nlin; ++ponto )
      {
         stack_top = -1;
         for( int i = size[ind] - 1; i >= 0; --i )
         {
            switch( phenotype[ind * data.size + i] )
            {
               #include <interpreter_core>
               case T_ATTRIBUTE:
                  stack[++stack_top] = data.inputs[ponto][(int)ephemeral[ind * data.size + i]];
                  //if ( ponto == 1 ) {printf( "T_ATTRIBUTE: %d %f \n", stack_top, stack[stack_top]);}
                  break;
#ifndef NOT_USING_T_CONST
               case T_CONST:
                  stack[++stack_top] = ephemeral[ind * data.size + i];
                  //if ( ponto == 1 ) {printf( "T_CONST: %d %f \n", stack_top, stack[stack_top]);}
                  break;
#endif
               default:
                  stack[++stack_top] = NAN; // "Invalidates" the stack (solution) if a non-recognized symbol (terminal) is given
                  break;
            }
         }
         if( ppp_mode && prediction_mode ) {
            vector[ponto] = stack[stack_top];
         }
         else {
            float error = ERROR(stack[stack_top], data.inputs[ponto][data.ncol-1]);

            // Avoid further calculations if the current one has overflown the float
            // (i.e., it is inf or NaN).
            if( std::isinf(error) || std::isnan(error) ) { sum = std::numeric_limits<float>::max(); break; }

#ifdef REDUCEMAX
            sum = (error*data.nlin > sum) ? error*data.nlin : sum;
#else
            sum += error;
#endif
         }
      }
      if( !prediction_mode )
      {
         if( std::isnan( sum ) || std::isinf( sum ) ) {vector[ind] = std::numeric_limits<float>::max();}
         else 
         {
            vector[ind] = sum/data.nlin + alpha * size[ind];
         }
      }
   }
#ifdef PROFILING
   data.time_total_kernel1  += t_kernel.elapsed();
   data.time_gen_kernel1     = t_kernel.elapsed();
   data.gpops_gen_kernel     = (sum_size_gen * data.nlin) / t_kernel.elapsed();
#endif

   if( !ppp_mode )
   {
#ifdef PROFILING
      util::Timer t_time;
#endif
      /* Reduction to find the best individuals. */
      util::PickNBest(*best_size, index, nInd, vector);
#ifdef PROFILING
      data.time_total_kernel2  += t_time.elapsed();
      data.time_gen_kernel2     = t_time.elapsed();
#endif
   }
}

void seq_interpret_destroy() 
{
   for( int i = 0; i < data.nlin; ++i )
     delete [] data.inputs[i];
   delete [] data.inputs;
}

#ifdef PROFILING
void seq_print_time( bool total, unsigned long long sum_size )
{ 
   double time_kernel1 = total ? data.time_total_kernel1 : data.time_gen_kernel1;
   double time_kernel2 = total ? data.time_total_kernel2 : data.time_gen_kernel2;
   double gpops_kernel = total ? (sum_size * data.nlin) / data.time_total_kernel1 : data.gpops_gen_kernel;

   printf(", time_kernel[1]: %lf, time_kernel[2]: %lf", time_kernel1, time_kernel2);
   printf("; gpops_kernel: %lf", gpops_kernel);
}
#endif
