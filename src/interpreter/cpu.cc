#include <stdio.h> 
#include <stdlib.h>
#include <cmath>    
#include <limits>
#include "cpu.h"


/** ****************************************************************** **/
/** ***************************** TYPES ****************************** **/
/** ****************************************************************** **/

static struct t_data { double** input; double** model; double* obs; unsigned size; int nlin; } data;


/** ****************************************************************** **/
/** ************************* MAIN FUNCTION ************************** **/
/** ****************************************************************** **/

void interpret_init( const unsigned size, double** input, double** model, double* obs, int nlin, int ninput, int nmodel ) 
{
   data.size = size;
   data.nlin = nlin;

   data.input = new double*[nlin];
   for( int i = 0; i < nlin; i++ )
     data.input[i] = new double[ninput];
   data.model = new double*[nlin];
   for( int i = 0; i < nlin; i++ )
     data.model[i] = new double[nmodel];
   data.obs = new double[nlin];

   for( int i = 0; i < nlin; i++ )
   {
     for( int j = 0; j < ninput; j++ )
     {
       data.input[i][j] = input[i][j];
     }
     for( int j = 0; j < nmodel; j++ )
     {
       data.model[i][j] = model[i][j];
     }
     data.obs[i] = obs[i];
   }
}

void interpret( Symbol* phenotype, double* ephemeral, int size, double* vector, int mode )
{
   double pilha[data.size]; 
   double error[data.nlin];
   double sum = 0.0; 
   double sumdiff = 0.0; 
   double max = std::numeric_limits<float>::min(); 
   double min = std::numeric_limits<float>::max();

   int topo;
   for( int ponto = 0; ponto < data.nlin; ++ponto )
   {
      topo = -1;
      for( int i = size - 1; i >= 0; --i )
      {
         switch( phenotype[i] )
         {
            case T_IF_THEN_ELSE:
               if( pilha[topo] == 1.0 ) { --topo; }
               else { topo = topo - 2; }
               break;
            case T_AND:
               if( pilha[topo] == 1.0 && pilha[topo - 1] == 1.0 ) { pilha[topo - 1] = 1.0; --topo; }
               else { pilha[topo - 1] = 0.0; --topo; }
               break;
            case T_OR:
               if( pilha[topo] == 1.0 || pilha[topo - 1] == 1.0 ) { pilha[topo - 1] = 1.0; --topo; }
               else { pilha[topo - 1] = 0.0; --topo; }
               break;
            case T_NOT:
               pilha[topo] == !pilha[topo];
               break;
            case T_MAIOR:
               if( pilha[topo] > pilha[topo - 1] ) { pilha[topo - 1] = 1.0; --topo; }
               else { pilha[topo - 1] = 0.0; --topo; }
               break;
            case T_MENOR:
               if( pilha[topo] < pilha[topo - 1] ) { pilha[topo - 1] = 1.0; --topo; }
               else { pilha[topo - 1] = 0.0; --topo; }
               break;
            case T_IGUAL:
               if( pilha[topo] == pilha[topo - 1] ) { pilha[topo - 1] = 1.0; --topo; }
               else { pilha[topo - 1] = 0.0; --topo; }
               break;
            case T_ADD:
               pilha[topo - 1] = pilha[topo] + pilha[topo - 1]; --topo;
               break;
            case T_SUB:
               pilha[topo - 1] = pilha[topo] - pilha[topo - 1]; --topo;
               break;
            case T_MULT:
               pilha[topo - 1] = pilha[topo] * pilha[topo - 1]; --topo;
               break;
            case T_DIV:
               pilha[topo - 1] = pilha[topo] / pilha[topo - 1]; --topo;
               break;
            case T_MEAN:
               pilha[topo - 1] = (pilha[topo] + pilha[topo - 1]) / 2; --topo;
               break;
            case T_MAX:
               pilha[topo - 1] = fmax(pilha[topo], pilha[topo - 1]); --topo;
               break;
            case T_MIN:
               pilha[topo - 1] = fmin(pilha[topo], pilha[topo - 1]); --topo;
               break;
            case T_ABS:
               pilha[topo] = fabs(pilha[topo]);
               break;
            case T_SQRT:
               pilha[topo] = sqrt(pilha[topo]);
               break;
            case T_POW2:
               pilha[topo] = pow(pilha[topo], 2);
               break;
            case T_POW3:
               pilha[topo] = pow(pilha[topo], 3);
               break;
            case T_NEG:
               pilha[topo] = -pilha[topo];
               break;
            case T_BMA:
               pilha[++topo] = data.input[ponto][0];
               break;
            case T_CHUVA_ONTEM:
               pilha[++topo] = data.input[ponto][1];
               break;
            case T_CHUVA_ANTEONTEM:
               pilha[++topo] = data.input[ponto][2];
               break;
            case T_MEAN_MODELO:
               pilha[++topo] = data.input[ponto][3];
               break;
            case T_MAX_MODELO:
               pilha[++topo] = data.input[ponto][4];
               break;
            case T_MIN_MODELO:
               pilha[++topo] = data.input[ponto][5];
               break;
            case T_STD_MODELO:
               pilha[++topo] = data.input[ponto][6];
               break;
            case T_CHOVE:
               pilha[++topo] = data.input[ponto][7];
               break;
            case T_CHUVA_LAG1P:
               pilha[++topo] = data.input[ponto][8];
               break;
            case T_CHUVA_LAG1N:
               pilha[++topo] = data.input[ponto][9];
               break;
            case T_CHUVA_LAG2P:
               pilha[++topo] = data.input[ponto][10];
               break;
            case T_CHUVA_LAG2N:
               pilha[++topo] = data.input[ponto][11];
               break;
            case T_CHUVA_LAG3P:
               pilha[++topo] = data.input[ponto][12];
               break;
            case T_CHUVA_LAG3N:
               pilha[++topo] = data.input[ponto][13];
               break;
            case T_PADRAO_MUDA:
               pilha[++topo] = data.input[ponto][14];
               break;
            case T_PERTINENCIA:
               pilha[++topo] = data.input[ponto][15];
               break;
            case T_CHUVA_PADRAO:
               pilha[++topo] = data.input[ponto][16];
               break;
            case T_CHUVA_HISTORICA:
               pilha[++topo] = data.input[ponto][17];
               break;
            case T_K:
               pilha[++topo] = data.input[ponto][18];
               break;
            case T_TT:
               pilha[++topo] = data.input[ponto][19];
               break;
            case T_SWEAT:
               pilha[++topo] = data.input[ponto][20];
               break;
            case T_PAD:
               pilha[++topo] = data.input[ponto][21];
               break;
            case T_MOD1:
               pilha[++topo] = data.model[ponto][0];
               break;
            case T_MOD2:
               pilha[++topo] = data.model[ponto][1];
               break;
            case T_MOD3:
               pilha[++topo] = data.model[ponto][2];
               break;
            case T_MOD4:
               pilha[++topo] = data.model[ponto][3];
               break;
            case T_MOD5:
               pilha[++topo] = data.model[ponto][4];
               break;
            case T_MOD6:
               pilha[++topo] = data.model[ponto][5];
               break;
            case T_MOD7:
               pilha[++topo] = data.model[ponto][6];
               break;
            case T_MOD8:
               pilha[++topo] = data.model[ponto][7];
               break;
            case T_1:
               pilha[++topo] = 1;
               break;
            case T_2:
               pilha[++topo] = 2;
               break;
            case T_3:
               pilha[++topo] = 3;
               break;
            case T_4:
               pilha[++topo] = 4;
               break;
            case T_EFEMERO:
               pilha[++topo] = ephemeral[i];
               break;
         }
      }
      if( mode ) {vector[ponto] = pilha[topo];}

      sum += fabs(pilha[topo] - data.obs[ponto]);

      error[ponto] = fabs(pilha[topo] - data.obs[ponto]);
      if( error[ponto] < min ) {min = error[ponto];}
      if( error[ponto] > max ) {max = error[ponto];}
   }

   if( mode )
   {
      if( isnan( sum ) || isinf( sum ) ) 
      {
         vector[data.nlin] = std::numeric_limits<float>::max();
         vector[data.nlin+1] = 0.0/0.0;
         vector[data.nlin+2] = 0.0/0.0;
         vector[data.nlin+3] = 0.0/0.0;
      } 
      else 
      {
         for( int i = 0; i < data.nlin; i++ ) {sumdiff += pow((error[i]-sum/data.nlin),2);}
         vector[data.nlin]   = sum/data.nlin;
         vector[data.nlin+1] = sqrt(sumdiff/(data.nlin-1));
         vector[data.nlin+2] = min;
         vector[data.nlin+3] = max;
      }
   }
   else
   {
      if( isnan( sum ) || isinf( sum ) ) 
      {
         vector[0] = std::numeric_limits<float>::max();
         vector[1] = 0.0/0.0;
         vector[2] = 0.0/0.0;
         vector[3] = 0.0/0.0;
      } 
      else 
      {
         for( int i = 0; i < data.nlin; i++ ) {sumdiff += pow((error[i]-sum/data.nlin),2);}
         vector[0] = sum/data.nlin;
         vector[1] = sqrt(sumdiff/(data.nlin-1));
         vector[2] = min;
         vector[3] = max;
      }
   }
}

void interpret_destroy() 
{
   delete[] data.obs;

   for( int i = 0; i < data.nlin; ++i )
     delete [] data.input[i];
   delete [] data.input;

   for( int i = 0; i < data.nlin; ++i )
     delete [] data.model[i];
   delete [] data.model;
}
