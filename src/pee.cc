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
#include <ctime>
#include <string>   
#include <sstream>
#include <iostream> 
#include "util/CmdLineParser.h"
#include "interpreter/accelerator.h"
#include "interpreter/sequential.h"
#include "pee.h"
#include "server/server.h"
#include "client/client.h"
#include "individual"
#include "grammar"

using namespace std;

/** ****************************************************************** **/
/** ***************************** TYPES ****************************** **/
/** ****************************************************************** **/

struct Peer {
  Peer( const std::string& s, float f ):
      address( s ), frequency( f ) {}

  std::string address;
  float frequency;
};

namespace { struct t_data { Symbol initial_symbol; Population best_individual; int best_size; unsigned max_size_phenotype; int nlin; Symbol* phenotype; float* ephemeral; int* size; int verbose; int elitism; int population_size; int immigrants_size; int generations; int number_of_bits; int bits_per_gene; int bits_per_constant; int seed; int tournament_size; float mutation_rate; float crossover_rate; float interval[2]; int version; double time_total; std::vector<Peer> peers; Pool* pool; } data; };

/** ****************************************************************** **/
/** *********************** AUXILIARY FUNCTIONS ********************** **/
/** ****************************************************************** **/

#define swap(i, j) {Population t = *i; *i = *j; *j = t;}

double random_number() {return (double)rand() / ((double)RAND_MAX + 1.0f);} // [0.0, 1.0)

t_rule* decode_rule( const int* genome, int* const allele, Symbol cabeca )
{
   // Verifica se é possível decodificar mais uma regra, caso contrário
   // retorna uma regra nula.
   if( *allele + data.bits_per_gene > data.number_of_bits ) { return NULL; }

   // Número de regras encabeçada por "cabeça"
   unsigned num_regras = tamanhos[cabeca];

   // Converte data.bits_per_gene bits em um valor integral
   unsigned valor_bruto = 0;
   for( int i = 0; i < data.bits_per_gene; ++i )  
      if( genome[(*allele)++] ) valor_bruto += pow( 2, i );

   // Seleciona uma regra no intervalo [0, num_regras - 1]
   return gramatica[cabeca][valor_bruto % num_regras];
}

float decode_real( const int* genome, int* const allele )
{
   // Retorna a constante zero se não há número suficiente de bits
   if( *allele + data.bits_per_constant > data.number_of_bits ) { return 0.; }

   // Converte data.bits_per_constant bits em um valor real
   float valor_bruto = 0.;
   for( int i = 0; i < data.bits_per_constant; ++i ) 
      if( genome[(*allele)++] ) valor_bruto += pow( 2.0, i );

   // Normalizar para o intervalo desejado: a + valor_bruto * (b - a)/(2^n - 1)
   return data.interval[0] + valor_bruto * (data.interval[1] - data.interval[0]) / 
          (pow( 2.0, data.bits_per_constant ) - 1);
}

int decode( const int* genome, int* const allele, Symbol* phenotype, float* ephemeral, int pos, Symbol initial_symbol )
{
   t_rule* r = decode_rule( genome, allele, initial_symbol ); 
   if( !r ) { return 0; }

   for( int i = 0; i < r->quantity; ++i )
      if( r->symbols[i] >= TERMINAL_MIN )
      {
         phenotype[pos] = r->symbols[i];

         // Tratamento especial para constantes efêmeras
         if( r->symbols[i] == T_CONST )
         {
           /* 
              Esta estratégia faz uso do próprio código genético do indivíduo 
              para extrair uma constante real. Extrai-se data.bits_per_constant bits a 
              partir da posição atual e os decodifica como um valor real.
            */
            ephemeral[pos] = decode_real( genome, allele );
         }
         else
         {
            if( r->symbols[i] >= ATTRIBUTE_MIN )
            {
               phenotype[pos] = T_ATTRIBUTE;
               ephemeral[pos] = r->symbols[i] - ATTRIBUTE_MIN;
            } 
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

void pee_init( float** input, int nlin, int argc, char** argv ) 
{
   CmdLine::Parser Opts( argc, argv );

   Opts.Bool.Add( "-v", "--verbose" );

   Opts.Bool.Add( "-acc" );

   Opts.String.Add( "-error", "--function-difference", "fabs((X)-(Y))" );

   Opts.Int.Add( "-ncol", "--number-of-columns" );

   Opts.Int.Add( "-g", "--generations", 50, 1, std::numeric_limits<int>::max() );

   Opts.Int.Add( "-s", "--seed", 0, 0, std::numeric_limits<long>::max() );

   Opts.Int.Add( "-ps", "--population-size", 1024, 1, std::numeric_limits<int>::max() );
   Opts.Int.Add( "-is", "--immigrants-size", 5, 1 );
   Opts.Float.Add( "-cp", "--crossover-probability", 0.95, 0.0, 1.0 );
   Opts.Float.Add( "-mr", "--mutation-rate", 0.01, 0.0, 1.0 );
   Opts.Int.Add( "-ts", "--tournament-size", 3, 1, std::numeric_limits<int>::max() );

   Opts.Bool.Add( "-e", "--elitism" );

   Opts.Int.Add( "-nb", "--number-of-bits", 2000, 16 );
   Opts.Int.Add( "-bg", "--bits-per-gene", 8, 8 );
   Opts.Int.Add( "-bc", "--bits-per-constant", 16, 4 );

   Opts.Float.Add( "-min", "--min-constant", 0 );
   Opts.Float.Add( "-max", "--max-constant", 300 );

   Opts.String.Add( "-peers", "--address_of_island" );

   // processing the command-line
   Opts.Process();

   // getting the results!
   data.verbose = Opts.Bool.Get("-v");

   data.generations = Opts.Int.Get("-g");

   data.seed = Opts.Int.Get("-s") == 0 ? time( NULL ) : Opts.Int.Get("-s");

   data.population_size = Opts.Int.Get("-ps");
   data.immigrants_size = Opts.Int.Get("-is");
   data.crossover_rate = Opts.Float.Get("-cp");
   data.mutation_rate = Opts.Float.Get("-mr");
   data.tournament_size = Opts.Int.Get("-ts");

   data.elitism = Opts.Bool.Get("-e");

   data.number_of_bits = Opts.Int.Get("-nb");
   data.bits_per_gene = Opts.Int.Get("-bg");
   data.bits_per_constant = Opts.Int.Get("-bc");

   if( Opts.Float.Get("-min") > Opts.Float.Get("-max") )
   {
      data.interval[0] = Opts.Float.Get("-max");
      data.interval[1] = Opts.Float.Get("-min");
   }
   else
   {
      data.interval[0] = Opts.Float.Get("-min");
      data.interval[1] = Opts.Float.Get("-max");
   }

   data.initial_symbol = NT_INICIAL;

   data.best_size = 1;
   data.best_individual.genome = NULL;
   data.best_individual.fitness = NULL;

   data.max_size_phenotype = MAX_QUANT_SIMBOLOS_POR_REGRA * data.number_of_bits/data.bits_per_gene;
   data.nlin = nlin;

   data.phenotype = new Symbol[data.population_size * data.max_size_phenotype];
   data.ephemeral = new float[data.population_size * data.max_size_phenotype];
   data.size = new int[data.population_size];

   std::string str = Opts.String.Get("-peers");
   //std::cout << str << std::endl;

   size_t pos;
   std::string s;
   float f;

   std::string delimiter = ",";
   while ( ( pos = str.find( delimiter ) ) != std::string::npos ) 
   {
      pos = str.find(delimiter);
      s = str.substr(0, pos);
      str.erase(0, pos + delimiter.length());
      //std::cout << s << std::endl;

      delimiter = ";";

      pos = str.find(delimiter);
      f = std::atof(str.substr(0, pos).c_str());
      str.erase(0, pos + delimiter.length());
      //std::cout << f << std::endl;

      data.peers.push_back( Peer( s, f ) );

      delimiter = ",";
   }
   
   //A thread pool used to manage the sending of individuals to the islands.
   data.pool = new Pool( data.peers.size() );

   {
      Poco::FastMutex::ScopedLock lock( Server::m_mutex );

      Server::m_immigrants = new std::vector<char>[data.immigrants_size];
      Server::m_fitness = new float[data.immigrants_size];
      for( int i = 0; i < data.immigrants_size; i++ )
      {
         Server::m_freeslots.push(i);
      }
   }

   data.version = Opts.Bool.Get("-acc");
   if( data.version )
   {
      if( acc_interpret_init( argc, argv, data.max_size_phenotype, data.population_size, input, nlin, 0 ) )
      {
         fprintf(stderr,"Error in initialization phase.\n");
      }
   }
   else
   {
      seq_interpret_init( Opts.String.Get("-error"), data.max_size_phenotype, input, nlin, Opts.Int.Get("-ncol") );
   }
}

void pee_clone( Population* original, int idx_original, Population* copy, int idx_copy )
{
   for( int i = 0; i < data.number_of_bits; ++i ) copy->genome[idx_copy * data.number_of_bits + i] = original->genome[idx_original * data.number_of_bits + i];

   copy->fitness[idx_copy] = original->fitness[idx_original];
}

void pee_individual_print( const Population* individual, int idx, FILE* out, int print_mode )
{
   Symbol phenotype[data.max_size_phenotype];
   float ephemeral[data.max_size_phenotype];

   int allele = 0;
   int size = decode( individual->genome + (idx * data.number_of_bits), &allele, phenotype, ephemeral, 0, data.initial_symbol );
   if( !size ) { return; }
  
   if( print_mode )
   {
      fprintf( out, "%d\n", size );
      for( int i = 0; i < size; ++i )
         if( phenotype[i] == T_CONST || phenotype[i] == T_ATTRIBUTE )
            fprintf( out, "%d %.12f ", phenotype[i], ephemeral[i] );
         else
            fprintf( out, "%d ", phenotype[i] );
      fprintf( out, "\n" );
   } 
   else 
      fprintf( out, "%d ", size );

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
            case T_GREATER:
               fprintf( out, "> " );
               break;
            case T_LESS:
               fprintf( out, "< " );
               break;
            case T_EQUAL:
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
            //case T_MOD:
            //   fprintf( out, "MOD " );
            //   break;
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
            case T_ATTRIBUTE:
               fprintf( out, "ATTR-%d ", (int)ephemeral[i] );
               break;
            case T_1P:
               fprintf( out, "1 " );
               break;
            case T_2P:
               fprintf( out, "2 " );
               break;
            case T_3P:
               fprintf( out, "3 " );
               break;
            case T_4P:
               fprintf( out, "4 " );
               break;
            case T_CONST:
               fprintf( out, "%.12f ",  ephemeral[i] );
               break;
      } 
   if( print_mode )
   {
      fprintf( out, "\n" );
      fprintf( out, "%.12f\n", individual->fitness[idx] );
   }
   else
      fprintf( out, " %.12f\n", individual->fitness[idx] );
}

int pee_tournament( const float* fitness )
{
   int idx_winner = (int)(random_number() * data.population_size);
   float fitness_winner = fitness[idx_winner];

   for( int t = 1; t < data.tournament_size; ++t )
   {
      int idx_competitor = (int)(random_number() * data.population_size);
      const float fitness_competitor = fitness[idx_competitor];

      if( fitness_competitor < fitness_winner ) 
      {
         fitness_winner = fitness_competitor;
         idx_winner = idx_competitor;
      }
   }

   return idx_winner;
}

void pee_send_individual( Population* population )
{
   for( int i = 0; i < data.peers.size(); i++ )
   { 
      if( random_number() < data.peers[i].frequency )
      {
         //Testa se o thread ainda está mandando o indivíduo, escolhido na geração anterior, para a ilha.
         //if( !(data.pool->threads[i] == NULL) && data.pool->threads[i]->isRunning() ) continue;
         if( data.pool->threads[i]->isRunning() ) continue;

         const int idx = pee_tournament( population->fitness );

         std::stringstream results; //results.str(std::string());
         results <<  population->fitness[idx] << " ";
         for( int j = 0; j < data.number_of_bits-1; j++ )
            results <<  population->genome[idx * data.number_of_bits + j] << " ";
         results <<  population->genome[idx * data.number_of_bits + data.number_of_bits];

         delete data.pool->clients[i], data.pool->ss[i];
         data.pool->ss[i] = new StreamSocket();
         data.pool->clients[i] = new Client( *(data.pool->ss[i]), data.peers[i].address.c_str(), results.str() );
         data.pool->threads[i]->start( *( data.pool->clients[i] ) );

         std::cerr << "Sending Individual Thread[" << i << "] to " << data.peers[i].address << ": " << population->fitness[idx] << std::endl;
         //std::cerr << results.str() << std::endl;
      }
   }
}

int pee_receive_individual( int* immigrants )
{
   int slot; int nImmigrants = 0;
   while( !Server::m_ready.empty() && nImmigrants < data.immigrants_size )
   {
      {
         Poco::FastMutex::ScopedLock lock( Server::m_mutex );

         slot = Server::m_ready.front();
         Server::m_ready.pop();
      }
      
      int offset;
      char *tmp = Server::m_immigrants[slot].data(); 

      sscanf( tmp, "%f%n", &Server::m_fitness[slot], &offset );
      tmp += offset + 1; 
      
      //std::cerr << "Receiving[slot=" << slot << "]: " << tmp << std::endl;

      //tmp += offset;
      //fprintf(stdout,"Receiving[slot=%d]: ",slot);
      //for( int i = 0; i < (data.number_of_bits - 1); i++ )
      //{
      //   sscanf( tmp, "%d%n", &immigrants[nImmigrants * data.number_of_bits + i], &offset );
      //   tmp += offset;
      //   fprintf(stdout,"%d ",immigrants[nImmigrants * data.number_of_bits + i]);
      //}
      //sscanf( tmp, "%d", &immigrants[nImmigrants * (data.number_of_bits - 1)] );
      //fprintf(stdout,"%d\n",immigrants[nImmigrants * (data.number_of_bits - 1)]);

      //fprintf(stdout,"Receiving[slot=%d]: ",slot);

      /* FIXME:
         Existe um caso que precisa ser tratado:
          - quando o tamanho do array apontado por 'tmp' é *menor* do que 'data.number_of_bits' -> neste caso só se pode copiar a quantidade de chars que existem em 'tmp' (é preciso verificar pelo caracter fim de linha, isto é, '\0').
      */
      for( int i = 0, j = 0; i < data.number_of_bits; i++, j += 2 )
      {
         immigrants[nImmigrants * data.number_of_bits + i] = tmp[j] - '0';
         //fprintf(stdout,"%d ",immigrants[nImmigrants * data.number_of_bits + i]);
      }
      nImmigrants++;
      //fprintf(stdout,"\n");

      {
         Poco::FastMutex::ScopedLock lock( Server::m_mutex );

         Server::m_freeslots.push(slot);
      }

   }
   return nImmigrants;
}

void pee_evaluate( Population* descendentes, Population* antecedentes, int* nImmigrants )
{
   int allele;
   for( int i = 0; i < data.population_size; i++ )
   {
      allele = 0;
      data.size[i] = decode( descendentes->genome + (i * data.number_of_bits), &allele, data.phenotype + (i * data.max_size_phenotype), data.ephemeral + (i * data.max_size_phenotype), 0, data.initial_symbol );
   }
   
//   for( int j = 0; j < nInd; j++ )
//   {
//      fprintf(stdout,"Ind[%d]=%d\n",j,data.size[j]);
//      for( int i = 0; i < data.size[j]; i++ )
//      {
//         fprintf(stdout,"%d ",data.phenotype[j * data.max_size_phenotype + i]);
//      }
//      fprintf(stdout,"\n");
//      for( int i = 0; i < data.size[j]; i++ )
//      {
//         fprintf(stdout,"%f ",data.ephemeral[j * data.max_size_phenotype + i]);
//      }
//      fprintf(stdout,"\n");
//   }

   
   int index[data.best_size];

   if( data.version )
   {
      acc_interpret( data.phenotype, data.ephemeral, data.size, descendentes->fitness, data.population_size, &pee_send_individual, &pee_receive_individual, antecedentes, nImmigrants, index, &data.best_size, 0, 0.00001 );
   }
   else
   {
      seq_interpret( data.phenotype, data.ephemeral, data.size, descendentes->fitness, data.population_size, index, &data.best_size, 0, 0.00001 );
      *nImmigrants = pee_receive_individual( antecedentes->genome );
   }
   //std::cout << data.best_size << std::endl;

   //bool flag = false;
   //for( int i = 0; i < nInd; i++ )
   //{
   //   //if( population[i].fitness < data.best_individual.fitness )
   //   //{
   //   //   flag = true;
   //   //   pee_clone( &population[i], &data.best_individual );
   //   //}
   //}
   
   //if( flag )
   //{
   //   printf("Método1:%d  ",flag);
   //   pee_individual_print( &data.best_individual, stdout, 0 );
   //   printf("Método2:%d  ",flag);
   //   pee_individual_print( &population[index], stdout, 0 );
   //}

   for( int i = 0; i < data.best_size; i++ )
   {
      if( descendentes->fitness[index[i]] < data.best_individual.fitness[i] )
      {
         pee_clone( descendentes, index[i], &data.best_individual, i );
      }
   }
}

void pee_generate_population( Population* antecedentes, Population* descendentes, int* nImmigrants )
{
   for( int i = 0; i < data.population_size; ++i)
   {
      for( int j = 0; j < data.number_of_bits; j++ )
      {
         antecedentes->genome[i * data.number_of_bits + j] = (random_number() < 0.5) ? 1 : 0;
      }
   }
   pee_evaluate( antecedentes, descendentes, nImmigrants );
}

void pee_crossover( const int* father, const int* mother, int* offspring1, int* offspring2 )
{
#ifdef TWO_POINT_CROSSOVER
   // Cruzamento de dois pontos
   int pontos[2];

   // Cortar apenas nas bordas dos genes
   pontos[0] = ((int)(random_number() * data.number_of_bits))/data.bits_per_gene * data.bits_per_gene;
   pontos[1] = ((int)(random_number() * data.number_of_bits))/data.bits_per_gene * data.bits_per_gene;

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
   for( int i = pontos[1]; i < data.number_of_bits; ++i )
   {
      offspring1[i] = father[i];
      offspring2[i] = mother[i];
   }
#else
   // Cruzamento de um ponto
   int pontoDeCruzamento = (int)(random_number() * data.number_of_bits);

   for( int i = 0; i < pontoDeCruzamento; ++i )
   {
      offspring1[i] = father[i];
      offspring2[i] = mother[i];
   }
   for( int i = pontoDeCruzamento; i < data.number_of_bits; ++i )
   {
      offspring1[i] = mother[i];
      offspring2[i] = father[i];
   }
#endif
}

void pee_mutation( int* genome )
{
   /* First the number of bit positions that will be mutated is chosen
    * (num_bits_mutated), based on 'mutation_rate', which defines the maximum
    * fraction of the vector of bits that can be mutated at once. */
   const int max_bits_mutated = ceil(data.mutation_rate * data.number_of_bits);
   int num_bits_mutated = (int) (random_number() * (max_bits_mutated+1));

   /* Then, each position is selected at random and its value is swapped. */
   while( num_bits_mutated-- > 0 )
   {
      int bit = (int)(random_number() * data.number_of_bits);
      genome[bit] = !genome[bit];
   }
}

void pee_print_best( FILE* out, int print_mode ) 
{
   pee_individual_print( &data.best_individual, 0, out, print_mode );
}

void pee_print_time() 
{
   if( data.version )
   {
      acc_print_time();
   }
   printf("time_total: %lf\n", data.time_total);
}

void pee_evolve()
{
   /*
      Pseudo-code for evolve:

   1: Create (randomly) the initial population P
   2: Evaluate all individuals (programs) of P
   3: for generation ← 1 to NG do
      4: Copy the best (elitism) individuals of P to the temporary population Ptmp
      5: while |Ptmp| < |P| do
         6: Select and copy from P two fit individuals, p1 and p2
         7: if [probabilistically] crossover then
            8: Recombine p1 and p2, creating the children p1' and p2'
            9: p1 ← p1' ; p2 ← p2'
         10: end if
         11: if [probabilistically] mutation then
            12: Apply mutation operators in p1 and p2, generating p1' and p2'
            13: p1 ← p1' ; p2 ← p2'
         14: end if
         15: Insert p1 and p2 into Ptmp
      16: end while
      17: Evaluate all individuals (programs) of Ptmp
      18: P ← Ptmp; then discard Ptmp
   19: end for
   20: return the best individual so far
   */

   clock_t start, end;
   start = clock();

   srand( data.seed );

   Population antecedentes, descendentes;

   antecedentes.fitness = new float[data.population_size];
   descendentes.fitness = new float[data.population_size];

   antecedentes.genome = new int[data.population_size * data.number_of_bits];
   descendentes.genome = new int[data.population_size * data.number_of_bits];

   data.best_individual.fitness = new float[data.best_size];
   data.best_individual.genome = new int[data.best_size * data.number_of_bits];
   for( int i = 0; i < data.best_size; i++ )
   {
      data.best_individual.fitness[i] = std::numeric_limits<float>::max();
   }

   int nImmigrants;

   clock_t start1, end1;
   start1 = clock();
   // 1 e 2:
   pee_generate_population( &antecedentes, &descendentes, &nImmigrants );
   end1 = clock();
   cerr << "Generate Population Time: " << ((double)(end1 - start1))/((double)(CLOCKS_PER_SEC)) << endl;

   double time_tournament = 0.;
   double time_crossover  = 0.;
   double time_mutation   = 0.;
   double time_clone      = 0.;
   
   // 3:
   for( int geracao = 1; geracao <= data.generations; ++geracao )
   {
      // 4:
      if( data.elitism ) 
      {
         pee_clone( &data.best_individual, 0, &descendentes, 0 );
         nImmigrants++;
      }

      //std::cerr << "nImmigrants[geration:" << geracao << "]: " << nImmigrants << std::endl;

      // 5
      for( int i = nImmigrants; i < data.population_size; i += 2 )
      {

         clock_t start2, end2;
         start2 = clock();
         // 6:
         int idx_father = pee_tournament( antecedentes.fitness );
         int idx_mother = pee_tournament( antecedentes.fitness );
         end2 = clock();
         time_tournament += ((double)(end2 - start2))/((double)(CLOCKS_PER_SEC));

         // 7:
         if( random_number() < data.crossover_rate )
         {
            clock_t start3, end3;
            start3 = clock();
            // 8 e 9:
            if( i < ( data.population_size - 1 ) )
            {
               pee_crossover( antecedentes.genome + (idx_father * data.number_of_bits), antecedentes.genome + (idx_mother * data.number_of_bits), descendentes.genome + (i * data.number_of_bits), descendentes.genome + ((i + 1) * data.number_of_bits));
            }
            else 
            {
               pee_crossover( antecedentes.genome + (idx_father * data.number_of_bits), antecedentes.genome + (idx_mother * data.number_of_bits), descendentes.genome + (i * data.number_of_bits), descendentes.genome + (i * data.number_of_bits));
            }
            end3 = clock();
            time_crossover += ((double)(end3 - start3))/((double)(CLOCKS_PER_SEC));
         } // 10
         else 
         {
            clock_t start5, end5;
            start5 = clock();
            // 9:
            pee_clone( &antecedentes, idx_father, &descendentes, i );
            if( i < ( data.population_size - 1 ) )
            {
               pee_clone( &antecedentes, idx_mother, &descendentes, i + 1 );
            }
            end5 = clock();
            time_clone += ((double)(end5 - start5))/((double)(CLOCKS_PER_SEC));
         } // 10

         clock_t start4, end4;
         start4 = clock();
         // 11, 12, 13, 14 e 15:
         pee_mutation( descendentes.genome + (i * data.number_of_bits) );
         if( i < ( data.population_size - 1 ) )
         {
            pee_mutation( descendentes.genome + ((i + 1) * data.number_of_bits) );
         }
         end4 = clock();
         time_mutation += ((double)(end4 - start4))/((double)(CLOCKS_PER_SEC));
      } // 16

      clock_t start0, end0;
      start0 = clock();
      // 17:
      pee_evaluate( &descendentes, &antecedentes, &nImmigrants );
      end0 = clock();
      cerr << "Evaluate Time: " << ((double)(end0 - start0))/((double)(CLOCKS_PER_SEC)) << endl;

      //pee_send_individual( &descendentes );

      // 18:
      swap( &antecedentes, &descendentes );

      if( data.verbose ) 
      {
         printf("[%d] ", geracao);
         pee_individual_print( &data.best_individual, 0, stdout, 0 );
      }
   } // 19

   // Clean up
   delete[] antecedentes.genome, antecedentes.fitness, descendentes.genome, descendentes.fitness; 

   end = clock();
   data.time_total = ((double)(end - start))/((double)(CLOCKS_PER_SEC));
   cerr << "Tournament time: " << time_tournament << endl;
   cerr << "Crossover time: " << time_crossover << endl;
   cerr << "Mutation time: " << time_mutation << endl;
   cerr << "Clone time: " << time_clone << endl;
}

void pee_destroy()
{
   delete[] data.best_individual.genome, data.best_individual.fitness, data.phenotype, data.ephemeral, data.size;
   delete[] Server::m_immigrants, Server::m_fitness;
   delete data.pool;
   if( !data.version ) {seq_interpret_destroy();}
}
