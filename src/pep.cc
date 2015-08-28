/************************************************************************/
/**     Name:Amanda Sabatini Dufek                                     **/
/**          Douglas Adriano Augusto            Date:01/06/2015        **/
/**     Parallel Ensemble Prediction (PEP)                             **/
/************************************************************************/

/** ****************************************************************** **/
/** *************************** OBJECTIVE **************************** **/
/** ****************************************************************** **/
/**                                                                    **/
/** ****************************************************************** **/

#include <stdio.h> 
#include <stdlib.h>
#include <cmath>    
#include <string>   
#include "util/CmdLineParser.h"
#include "interpreter/accelerator.h"
#include "interpreter/sequential.h"
#include "pep.h"


/** ****************************************************************** **/
/** ***************************** TYPES ****************************** **/
/** ****************************************************************** **/

namespace { static struct t_data { int nlin; Symbol* phenotype; float* ephemeral; int* size; float* vector; int prediction; int version; } data; };

/** ****************************************************************** **/
/** ************************* MAIN FUNCTIONS ************************* **/
/** ****************************************************************** **/

void pep_init( float** input, float** model, float* obs, int nlin, int argc, char **argv ) 
{
   CmdLine::Parser Opts( argc, argv );

   Opts.String.Add( "-run", "--program_file" );
   Opts.Bool.Add( "-acc" );
   Opts.Bool.Add( "-pred", "--prediction" );
   Opts.Int.Add( "-ni", "--number_of_inputs" );
   Opts.Int.Add( "-nm", "--number_of_models" );
   Opts.Process();
   const char* file = Opts.String.Get("-run").c_str();

   FILE *arqentra;
   arqentra = fopen(file,"r");
   if(arqentra == NULL) 
   {
     printf("Nao foi possivel abrir o arquivo de entrada.\n");
     exit(-1);
   }
   
   data.prediction = Opts.Bool.Get("-pred");
   data.version = Opts.Bool.Get("-acc");

   data.size = new int[1];
   fscanf(arqentra,"%d",&data.size[0]);
   //fprintf(stdout,"%d\n",data.size[0]);

   data.phenotype = new Symbol[data.size[0]];
   data.ephemeral = new float[data.size[0]];

   for( int tmp, i = 0; i < data.size[0]; ++i )
   {
      fscanf(arqentra,"%d ",&tmp); data.phenotype[i] = (Symbol)tmp;
      //fprintf(stdout,"%d ",data.phenotype[i]);
      if( data.phenotype[i] == T_CONST || data.phenotype[i] == T_ATTRIBUTE )
      {
         fscanf(arqentra,"%f ",&data.ephemeral[i]);
         //fprintf(stdout,"%.12f ",data.ephemeral[i]);
      }
   }
   //fprintf(stdout,"\n");
   fclose (arqentra);

   data.nlin = nlin;

   if( data.version )
   {
      if( acc_interpret_init( argc, argv, data.size[0], 1, input, model, obs, nlin, data.prediction ) )
      {
         fprintf(stderr,"Error in initialization phase.\n");
      }
   }
   else
   {
      seq_interpret_init( data.size[0], input, model, obs, nlin, Opts.Int.Get("-ni"), Opts.Int.Get("-nm") );
   }
}

void pep_interpret()
{
   if( data.prediction )
   {
      data.vector = new float[data.nlin];
      if( data.version )
      {
         acc_interpret( data.phenotype, data.ephemeral, data.size, data.vector, 1, NULL, NULL, 1, 0 );
      }
      else
      {
         seq_interpret( data.phenotype, data.ephemeral, data.size, data.vector, 1, NULL, NULL, 1, 0 );
      }
   }
   else
   {
      data.vector = new float[1];
      if( data.version )
      {
         acc_interpret( data.phenotype, data.ephemeral, data.size, data.vector, 1, NULL, NULL, 0, 0 );
      }
      else
      {
         seq_interpret( data.phenotype, data.ephemeral, data.size, data.vector, 1, NULL, NULL, 0, 0 );
      }
   }
}

void pep_print( FILE* out )
{
   if( data.prediction )
   {
      for( int i = 0; i < data.nlin; ++i )
         fprintf( out, "%.12f\n", data.vector[i] );
   }
   else
      fprintf( out, "%.12f\n", data.vector[0] );
}

void pep_destroy() 
{
   delete[] data.phenotype, data.ephemeral, data.size, data.vector;
   if( !data.version ) {seq_interpret_destroy();}
}
