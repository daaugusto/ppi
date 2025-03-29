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
#include <cmath>    
#include <string>   
#include "util/CmdLineParser.h"
#include "interpreter/accelerator.h"
#include "interpreter/sequential.h"
#include "ppp.h"


/** ****************************************************************** **/
/** ***************************** TYPES ****************************** **/
/** ****************************************************************** **/

namespace ppi { static struct t_data { int nlin; Symbol* phenotype; float* ephemeral; int* size; float* vector; int prediction; int parallel_version; } data; };

/** ****************************************************************** **/
/** ************************* MAIN FUNCTIONS ************************* **/
/** ****************************************************************** **/

namespace ppi {

void ppp_init( float** input, int nlin, int ncol, int argc, char** argv ) 
{
   CmdLine::Parser Opts( argc, argv );

   Opts.String.Add( "-sol", "--solution" );
   Opts.Bool.Add( "-acc" );
   Opts.Bool.Add( "-pred", "--prediction" );
   Opts.Process();

   std::istringstream iss;
   std::string solution = Opts.String.Get("-sol");
   iss.str (solution);

   data.prediction = Opts.Bool.Get("-pred");
   data.parallel_version = Opts.Bool.Get("-acc");

   data.size = new int[1];
   // Put the size of the solution (number of Symbols) into the variable data.size[0]
   iss >> data.size[0];

   data.phenotype = new Symbol[data.size[0]];
   data.ephemeral = new float[data.size[0]];

   int actual_size = 0; int tmp = -1;
   for( int i = 0; i < data.size[0]; ++i )
   {
      iss >> tmp;
      if (iss)
         ++actual_size;
      else
         break; // Stops if the actual number of Symbols is less than the informed

      data.phenotype[i] = (Symbol)tmp;
      if(
#ifndef NOT_USING_T_CONST
      data.phenotype[i] == T_CONST ||
#endif
      data.phenotype[i] == T_ATTRIBUTE )
      {
         iss >> data.ephemeral[i];
      }
   }

   while (iss >> tmp) ++actual_size; // Let's check if there are still more Symbols remaining
   if (actual_size != data.size[0])
      std::cerr << "WARNING: The given size (" << data.size[0] << ") and actual size (" << actual_size << ") differ: '" << solution << "'\n";

   data.nlin = nlin;

   if( data.parallel_version )
   {
      if( acc_interpret_init( argc, argv, data.size[0], -1, 1, input, nlin, ncol, 1, data.prediction ) )
      {
         fprintf(stderr,"Error in initialization phase.\n");
      }
   }
   else
   {
      seq_interpret_init( data.size[0], input, nlin, ncol );
   }
}

void ppp_interpret()
{
   if( data.prediction )
   {
      data.vector = new float[data.nlin];
      if( data.parallel_version )
      {
         acc_interpret( data.phenotype, data.ephemeral, data.size, 
#ifdef PROFILING
         0,
#endif
         data.vector, 1, NULL, NULL, NULL, NULL, NULL, NULL, 1, 1, 0 );
      }
      else
      {
         seq_interpret( data.phenotype, data.ephemeral, data.size, 
#ifdef PROFILING
         0,
#endif
         data.vector, 1, NULL, NULL, 1, 1, 0 );
      }
   }
   else
   {
      data.vector = new float[1];
      if( data.parallel_version )
      {
         acc_interpret( data.phenotype, data.ephemeral, data.size,
#ifdef PROFILING
         0,
#endif
         data.vector, 1, NULL, NULL, NULL, NULL, NULL, NULL, 1, 0, 0 );
      }
      else
      {
         seq_interpret( data.phenotype, data.ephemeral, data.size, 
#ifdef PROFILING
         0,
#endif
         data.vector, 1, NULL, NULL, 1, 0, 0 );
      }
   }
}

void ppp_print( FILE* out )
{
   if( data.prediction )
   {
      for( int i = 0; i < data.nlin; ++i )
         fprintf( out, "%.12f\n", data.vector[i] );
   }
   else
      fprintf( out, "%.12f\n", data.vector[0] );
}

void ppp_destroy() 
{
   delete[] data.phenotype;
   delete[] data.ephemeral;
   delete[] data.size;
   delete[] data.vector;

   if( !data.parallel_version ) {seq_interpret_destroy();}
}

}
