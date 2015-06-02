/************************************************************************/
/**     Name:Amanda Sabatini Dufek                                     **/
/**          Douglas Adriano Augusto            Date:01/06/2015        **/
/**     Parallel Ensemble Evolution (PEE)                              **/
/************************************************************************/

/** ****************************************************************** **/
/** *************************** OBJECTIVE **************************** **/
/** ****************************************************************** **/
/**                                                                    **/
/** ****************************************************************** **/

#include <stdio.h> 
#include <stdlib.h>
#include <cmath>    
#include <limits>
#include "pee.h"
#include "grammar"
#include "parametros"

#define BITS_BY_GENE 8
#define BITS_BY_CONSTANT 16


/** ****************************************************************** **/
/** ***************************** TYPES ****************************** **/
/** ****************************************************************** **/

struct t_individual { int* genome; double fitness; };


/** ****************************************************************** **/
/** ************************ GLOBAL VARIABLES ************************ **/
/** ****************************************************************** **/

Symbol simboloInicial = NT_IF_THEN_ELSE_INICIAL;

Individual melhorIndividuo = { NULL, std::numeric_limits<float>::max()};

const unsigned max_size_phenotype = MAX_QUANT_SIMBOLOS_POR_REGRA * bits_number/BITS_BY_GENE;


/** ****************************************************************** **/
/** *********************** AUXILIARY FUNCTIONS ********************** **/
/** ****************************************************************** **/

#define swap(i, j) {Individual* t = i; i = j; j = t;}

double random_number() {return rand() / (RAND_MAX + 1.0);} // [0.0, 1.0)

t_rule* decode_rule( const int* genome, int* const allele, Symbol cabeca )
{
   // Verifica se é possível decodificar mais uma regra, caso contrário
   // retorna uma regra nula.
   if( *allele + BITS_BY_GENE > bits_number ) { return NULL; }

   // Número de regras encabeçada por "cabeça"
   unsigned num_regras = tamanhos[cabeca];

   // Converte BITS_BY_GENE bits em um valor integral
   unsigned valor_bruto = 0;
   for( int i = 0; i < BITS_BY_GENE; ++i )  
      if( genome[(*allele)++] ) valor_bruto += pow( 2, i );

   // Seleciona uma regra no intervalo [0, num_regras - 1]
   return gramatica[cabeca][valor_bruto % num_regras];
}

double decode_real( const int* genome, int* const allele )
{
   // Retorna a constante zero se não há número suficiente de bits
   if( *allele + BITS_BY_CONSTANT > bits_number ) { return 0.; }

   // Converte BITS_BY_CONSTANT bits em um valor real
   double valor_bruto = 0.;
   for( int i = 0; i < BITS_BY_CONSTANT; ++i ) 
      if( genome[(*allele)++] ) valor_bruto += pow( 2.0, i );

   // Normalizar para o intervalo desejado: a + valor_bruto * (b - a)/(2^n - 1)
   return interval[0] + valor_bruto * (interval[1] - interval[0]) / 
          (pow( 2.0, BITS_BY_CONSTANT ) - 1);
}

int decode( const int* genome, int* const allele, Symbol* phenotype, double* ephemeral, int pos, Symbol simbolo_inicial )
{
   t_rule* r = decode_rule( genome, allele, simbolo_inicial ); 
   if( !r ) { return 0; }

   for( int i = 0; i < r->quantity; ++i )
      if( r->symbols[i] >= TERMINAL_MIN )
      {
         phenotype[pos] = r->symbols[i];

         // Tratamento especial para constantes efêmeras
         if( r->symbols[i] == T_EFEMERO )
         {
           /* 
              Esta estratégia faz uso do próprio código genético do indivíduo 
              para extrair uma constante real. Extrai-se BITS_BY_CONSTANT bits a 
              partir da posição atual e os decodifica como um valor real.
            */
            ephemeral[pos] = decode_real( genome, allele );
         }
         ++pos;
      }
      else
      {
         pos = decode( genome, allele, phenotype, ephemeral, pos, r->symbols[i] );
         if( !pos ) return 0;
      }

   return pos;
}


/** ****************************************************************** **/
/** ************************* MAIN FUNCTIONS ************************* **/
/** ****************************************************************** **/

void evaluate( Individual* individual, double **input, double **model, double *obs, int start, int end )
{
   Symbol phenotype[max_size_phenotype];
   double  ephemeral[max_size_phenotype];

   int allele = 0;
   int size = decode( individual->genome, &allele, phenotype, ephemeral, 0, simboloInicial );
   if( !size ) { individual->fitness = std::numeric_limits<float>::max(); return; }

   double pilha[max_size_phenotype];
   double erro = 0.0; double soma = 0.0;
   int topo;
   for( int ponto = start; ponto <= end; ++ponto )
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
               pilha[++topo] = input[ponto][0];
               break;
            case T_CHUVA_ONTEM:
               pilha[++topo] = input[ponto][1];
               break;
            case T_CHUVA_ANTEONTEM:
               pilha[++topo] = input[ponto][2];
               break;
            case T_MEAN_MODELO:
               pilha[++topo] = input[ponto][3];
               break;
            case T_MAX_MODELO:
               pilha[++topo] = input[ponto][4];
               break;
            case T_MIN_MODELO:
               pilha[++topo] = input[ponto][5];
               break;
            case T_STD_MODELO:
               pilha[++topo] = input[ponto][6];
               break;
            case T_CHOVE:
               pilha[++topo] = input[ponto][7];
               break;
            case T_CHUVA_LAG1P:
               pilha[++topo] = input[ponto][8];
               break;
            case T_CHUVA_LAG1N:
               pilha[++topo] = input[ponto][9];
               break;
            case T_CHUVA_LAG2P:
               pilha[++topo] = input[ponto][10];
               break;
            case T_CHUVA_LAG2N:
               pilha[++topo] = input[ponto][11];
               break;
            case T_CHUVA_LAG3P:
               pilha[++topo] = input[ponto][12];
               break;
            case T_CHUVA_LAG3N:
               pilha[++topo] = input[ponto][13];
               break;
            case T_PADRAO_MUDA:
               pilha[++topo] = input[ponto][14];
               break;
            case T_PERTINENCIA:
               pilha[++topo] = input[ponto][15];
               break;
            case T_CHUVA_PADRAO:
               pilha[++topo] = input[ponto][16];
               break;
            case T_CHUVA_HISTORICA:
               pilha[++topo] = input[ponto][17];
               break;
            case T_K:
               pilha[++topo] = input[ponto][18];
               break;
            case T_TT:
               pilha[++topo] = input[ponto][19];
               break;
            case T_SWEAT:
               pilha[++topo] = input[ponto][20];
               break;
            case T_PAD:
               pilha[++topo] = input[ponto][21];
               break;
            case T_MOD1:
               pilha[++topo] = model[ponto][0];
               break;
            case T_MOD2:
               pilha[++topo] = model[ponto][1];
               break;
            case T_MOD3:
               pilha[++topo] = model[ponto][2];
               break;
            case T_MOD4:
               pilha[++topo] = model[ponto][3];
               break;
            case T_MOD5:
               pilha[++topo] = model[ponto][4];
               break;
            case T_MOD6:
               pilha[++topo] = model[ponto][5];
               break;
            case T_MOD7:
               pilha[++topo] = model[ponto][6];
               break;
            case T_MOD8:
               pilha[++topo] = model[ponto][7];
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
      erro += fabs(pilha[topo] - obs[ponto]); 
   }

   if( isnan( erro ) || isinf( erro ) ) { individual->fitness = std::numeric_limits<float>::max(); return; } 

   individual->fitness = erro/(end-start+1) + 0.00001*size; 

   if( individual->fitness < melhorIndividuo.fitness )
   {
      clone( individual, &melhorIndividuo );
   }
}

void individual_print( const Individual* individual, FILE* out )
{
   Symbol phenotype[max_size_phenotype];
   double  ephemeral[max_size_phenotype];

   int allele = 0;
   int size = decode( individual->genome, &allele, phenotype, ephemeral, 0, simboloInicial );
   if( !size ) { return; }

   fprintf( out, "{%d} ", size );
   for( int i = 0; i < size; ++i )
      switch( phenotype[i] )
      {
            case T_IF_THEN_ELSE:
               fprintf( out, "IF-THEN-ELSE " );
               break;
            case T_AND:
               fprintf( out, "AND " );
               break;
            case T_OR:
               fprintf( out, "OR " );
               break;
            case T_NOT:
               fprintf( out, "NOT " );
               break;
            case T_MAIOR:
               fprintf( out, "> " );
               break;
            case T_MENOR:
               fprintf( out, "< " );
               break;
            case T_IGUAL:
               fprintf( out, "= " );
               break;
            case T_ADD:
               fprintf( out, "+ " );
               break;
            case T_SUB:
               fprintf( out, "- " );
               break;
            case T_MULT:
               fprintf( out, "* " );
               break;
            case T_DIV:
               fprintf( out, "/ " );
               break;
            case T_MEAN:
               fprintf( out, "MEAN " );
               break;
            case T_MAX:
               fprintf( out, "MAX " );
               break;
            case T_MIN:
               fprintf( out, "MIN " );
               break;
            case T_ABS:
               fprintf( out, "ABS " );
               break;
            case T_SQRT:
               fprintf( out, "SQRT " );
               break;
            case T_POW2:
               fprintf( out, "POW2 " );
               break;
            case T_POW3:
               fprintf( out, "POW3 " );
               break;
            case T_NEG:
               fprintf( out, "NEG " );
               break;
            case T_BMA:
               fprintf( out, "BMA " );
               break;
            case T_CHUVA_ONTEM:
               fprintf( out, "CHUVA_ONTEM " );
               break;
            case T_CHUVA_ANTEONTEM:
               fprintf( out, "CHUVA_ANTEONTEM " );
               break;
            case T_MEAN_MODELO:
               fprintf( out, "MEAN_MODELO " );
               break;
            case T_MAX_MODELO:
               fprintf( out, "MAX_MODELO " );
               break;
            case T_MIN_MODELO:
               fprintf( out, "MIN_MODELO " );
               break;
            case T_STD_MODELO:
               fprintf( out, "STD_MODELO " );
               break;
            case T_CHOVE:
               fprintf( out, "CHOVE " );
               break;
            case T_CHUVA_LAG1P:
               fprintf( out, "CHUVA_LAG1P " );
               break;
            case T_CHUVA_LAG1N:
               fprintf( out, "CHUVA_LAG1N " );
               break;
            case T_CHUVA_LAG2P:
               fprintf( out, "CHUVA_LAG2P " );
               break;
            case T_CHUVA_LAG2N:
               fprintf( out, "CHUVA_LAG2N " );
               break;
            case T_CHUVA_LAG3P:
               fprintf( out, "CHUVA_LAG3P " );
               break;
            case T_CHUVA_LAG3N:
               fprintf( out, "CHUVA_LAG3N " );
               break;
            case T_PADRAO_MUDA:
               fprintf( out, "PADRAO_MUDA " );
               break;
            case T_PERTINENCIA:
               fprintf( out, "PERTINENCIA " );
               break;
            case T_CHUVA_PADRAO:
               fprintf( out, "CHUVA_PADRAO " );
               break;
            case T_CHUVA_HISTORICA:
               fprintf( out, "CHUVA_HISTORICA " );
               break;
            case T_K:
               fprintf( out, "K " );
               break;
            case T_TT:
               fprintf( out, "TT " );
               break;
            case T_SWEAT:
               fprintf( out, "SWEAT " );
               break;
            case T_PAD:
               fprintf( out, "PAD " );
               break;
            case T_MOD1:
               fprintf( out, "MOD " );
               break;
            case T_MOD2:
               fprintf( out, "MOD2 " );
               break;
            case T_MOD3:
               fprintf( out, "MOD3 " );
               break;
            case T_MOD4:
               fprintf( out, "MOD4 " );
               break;
            case T_MOD5:
               fprintf( out, "MOD5 " );
               break;
            case T_MOD6:
               fprintf( out, "MOD6 " );
               break;
            case T_MOD7:
               fprintf( out, "MOD7 " );
               break;
            case T_MOD8:
               fprintf( out, "MOD8 " );
               break;
            case T_1:
               fprintf( out, "1 " );
               break;
            case T_2:
               fprintf( out, "2 " );
               break;
            case T_3:
               fprintf( out, "3 " );
               break;
            case T_4:
               fprintf( out, "4 " );
               break;
            case T_EFEMERO:
               fprintf( out, "%.12f ",  ephemeral[i] );
               break;
      } 
   fprintf( out, " [Aptidao: %.12f]\n", individual->fitness );
}

void generate_population( Individual* population, double **input, double **model, double *obs, int start, int end )
{
   for( int i = 0; i < population_size; ++i)
   {
      for( int j = 0; j < bits_number; j++ )
      {
         population[i].genome[j] = (random_number() < 0.5) ? 1 : 0;
      }
      
      evaluate( &population[i], input, model, obs, start, end );
   }
}

void crossover( const int* father, const int* mother, int* offspring1, int* offspring2 )
{
#ifdef TWO_POINT_CROSSOVER
   // Cruzamento de dois pontos
   int pontos[2];

   // Cortar apenas nas bordas dos genes
   pontos[0] = ((int)(random_number() * bits_number))/BITS_BY_GENE * BITS_BY_GENE;
   pontos[1] = ((int)(random_number() * bits_number))/BITS_BY_GENE * BITS_BY_GENE;

   int tmp;
   if( pontos[0] > pontos[1] ) { tmp = pontos[0]; pontos[0] = pontos[1]; pontos[1] = tmp; }

   for( int i = 0; i < pontos[0]; ++i )
   {
      offspring1[i] = father[i];
      offspring2[i] = mother[i];
   }
   for( int i = pontos[0]; i < pontos[1]; ++i )
   {
      offspring1[i] = mother[i];
      offspring2[i] = father[i];
   }
   for( int i = pontos[1]; i < bits_number; ++i )
   {
      offspring1[i] = father[i];
      offspring2[i] = mother[i];
   }
#else
   // Cruzamento de um ponto
   int pontoDeCruzamento = (int)(random_number() * bits_number);

   for( int i = 0; i < pontoDeCruzamento; ++i )
   {
      offspring1[i] = father[i];
      offspring2[i] = mother[i];
   }
   for( int i = pontoDeCruzamento; i < bits_number; ++i )
   {
      offspring1[i] = mother[i];
      offspring2[i] = father[i];
   }
#endif
}

void mutation( int* genome )
{
   for( int i = 0; i < bits_number; ++i )
      if( random_number() < mutation_rate ) genome[i] = !genome[i];
}

void clone( const Individual* original, Individual* copy )
{
   for( int i = 0; i < bits_number; ++i ) copy->genome[i] = original->genome[i];

   copy->fitness = original->fitness;
}

const Individual* tournament( const Individual* population )
{
   const Individual* vencedor = &population[(int)(random_number() * population_size)];

   for( int t = 1; t < tournament_size; ++t )
   {
      const Individual* competidor = &population[(int)(random_number() * population_size)];

      if( competidor->fitness < vencedor->fitness ) vencedor = competidor;
   }

   return vencedor;
}

Individual evolve( double **input, double **model, double *obs, int start, int end )
{
   Individual* populacao_a = new Individual[population_size];
   Individual* populacao_b = new Individual[population_size];

   melhorIndividuo.genome = new int[bits_number];

   // Alocação dos indivíduos
   for( int i = 0; i < population_size; ++i )
   {
      populacao_a[i].genome = new int[bits_number];
      populacao_b[i].genome = new int[bits_number];
   }

   Individual* antecedentes = populacao_a;
   Individual* descendentes = populacao_b;

   // Criação da população inicial (1ª geração)
   generate_population( antecedentes, input, model, obs, start, end );
    
   // Processo evolucionário
   for( int geracao = 1; geracao <= generation_number && melhorIndividuo.fitness > 0.0005; ++geracao )
   {
      for( int i = 0; i < population_size; i += 2 )
      {
         // Seleção de indivíduos adaptados para gerar descendentes
         const Individual* father = tournament( antecedentes );
         const Individual* mother = tournament( antecedentes );

         // Cruzamento
         if( random_number() < crossover_rate )
         {
            crossover( father->genome, mother->genome, 
                        descendentes[i].genome, descendentes[i + 1].genome );
         }
         else // Apenas clonagem
         {
            clone( father, &descendentes[i] );
            clone( mother, &descendentes[i + 1] );
         }

         // Mutações
         mutation( descendentes[i].genome );
         mutation( descendentes[i + 1].genome );

         // Avaliação dos novos indivíduos
          evaluate( &descendentes[i], input, model, obs, start, end );
          evaluate( &descendentes[i + 1], input, model, obs, start, end );
      }

      // Elitismo
      if( elitism ) clone( &melhorIndividuo, &descendentes[0] );

      // Faz população nova ser a atual, e vice-versa.
      swap( antecedentes, descendentes );

   }

   // -----------------------
   // Liberação de memória
   // -----------------------
   for( int i = 0; i < population_size; ++i )
   {
      delete[] populacao_a[i].genome, populacao_b[i].genome;
   }
   delete[] populacao_a, populacao_b; 

   return melhorIndividuo;

   delete[] melhorIndividuo.genome;
}

