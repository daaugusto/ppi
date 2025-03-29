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

/* The probability of applying two-point crossover instead of one-point */
#define TWOPOINT_CROSSOVER_PROBABILITY 0.66

/* Probability of applying the standard mutation (bit flip) instead of shrink
 * mutation */
#define BITFLIP_MUTATION_PROBABILITY 0.75

/* When applying the shrink mutation, the variable
 * AGGRESSIVE_SHRINK_MUTATION_PROBABILITY defines the ratio in which the
 * aggressive version will be applied instead. The aggressive version ignores
 * the mutation rate and might even shrink the entire genome. */
#define AGGRESSIVE_SHRINK_MUTATION_PROBABILITY 0.10

#include <stdio.h> 
#include <stdlib.h>
#include <cmath>    
#include <limits>
#include <ctime>
#include <string>   
#include <sstream>
#include <iostream> 
#include "common/common.h"
#include "util/CmdLineParser.h"
#include "interpreter/accelerator.h"
#include "interpreter/sequential.h"
#include "ppi.h"
#include "server/server.h"
#include "client/client.h"
#include "individual"
#include "grammar"
#include "util/Util.h"
#include "util/Random.h"
#include "Poco/Logger.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// Definition of the Random Number Generator to be used (see util/Random.h)
//typedef Random RNG;
typedef XorShift128Plus RNG;

/*
 * The parameter ALPHA is the complexity penalization factor. Each individual
 * will have its error augmented by ALPHA*size, where size is the number of
 * operators and operands. ALPHA is usually very small, just enough to favor
 * those that have similar (same) error but are less complex than another.
 */
#define ALPHA 0.000001

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

namespace ppi { struct t_data { Symbol initial_symbol; Population best_individual; int best_size; unsigned max_size_phenotype; int nlin; Symbol* phenotype; float* ephemeral; int* size; unsigned long long sum_size; int verbose; int machine; int elitism; int population_size; int immigrants_size; int generations; int number_of_bits; int bits_per_gene; int bits_per_constant; int seed; int tournament_size; float mutation_rate; float crossover_rate; float interval[2]; int parallel_version; double time_total_evolve; double time_gen_evolve; double time_generate; double time_total_evaluate; double time_gen_evaluate; double gpops_gen_evaluate; double time_total_crossover; double time_gen_crossover; double time_total_mutation; double time_gen_mutation; double time_total_clone; double time_gen_clone; double time_total_tournament; double time_gen_tournament; double time_total_send; double time_total_receive; double time_gen_receive; double time_total_decode; double time_gen_decode; std::vector<Peer> peers; Pool* pool; unsigned long stagnation_tolerance; RNG ** RNGs; int argc; char ** argv;  } data; };

namespace ppi {

inline RNG * GetRNG() {
#ifdef _OPENMP
   return data.RNGs[omp_get_thread_num()];
#else
   return data.RNGs[0];
#endif
}

inline int GetMaxNumThreads() {
#ifdef _OPENMP
   return omp_get_max_threads();
#else
   return 1;
#endif
}

/** ****************************************************************** **/
/** *********************** AUXILIARY FUNCTIONS ********************** **/
/** ****************************************************************** **/

#define swap(i, j) {Population t = *i; *i = *j; *j = t;}

double random_number() { return GetRNG()->Real(); }
//double random_number() { double value = GetRNG()->Real(); std::cerr << value << std::endl; return value; }

t_rule* decode_rule( const GENOME_TYPE* genome, int* const allele, Symbol cabeca )
{
   // Verifica se é possível decodificar mais uma regra, caso contrário
   // retorna uma regra nula.
   if( *allele + data.bits_per_gene > data.number_of_bits ) { return NULL; }

   // Número de regras encabeçada por "cabeça"
   unsigned num_regras = tamanhos[cabeca];

   // Converte data.bits_per_gene bits em um valor integral
   unsigned valor_bruto = 0;
   for( int i = 0; i < data.bits_per_gene; ++i )
      if( genome[(*allele)++] ) valor_bruto += (1 << i); // (1<<i) is a more efficient way of performing 2^i

   // Seleciona uma regra no intervalo [0, num_regras - 1]
   return gramatica[cabeca][valor_bruto % num_regras];
}

float decode_real( const GENOME_TYPE* genome, int* const allele )
{
   // Retorna a constante zero se não há número suficiente de bits
   if( *allele + data.bits_per_constant > data.number_of_bits ) { return 0.; }

   // Converte data.bits_per_constant bits em um valor real
   // NB: This works backwards in the sense that the most significant bit is the last bit (from left to right).
   unsigned long valor_bruto = 0;
   for( int i = 0; i < data.bits_per_constant; ++i )
      if( genome[(*allele)++] ) valor_bruto += (1UL << i);

   // Normalizar para o intervalo desejado: a + valor_bruto * (b - a)/(2^n - 1)
   return data.interval[0] + float(valor_bruto) * (data.interval[1] - data.interval[0]) / ((1UL << data.bits_per_constant) - 1.0);
}

int decode( const GENOME_TYPE* genome, int* const allele, Symbol* phenotype, float* ephemeral, int pos, Symbol initial_symbol )
{
   t_rule* r = decode_rule( genome, allele, initial_symbol ); 
   if( !r || pos >= data.max_size_phenotype ) { return 0; } /* When setting max_size_phenotype (via -mps) to a value less than
                                                               what would be required (the true max size phenotype), it might
                                                               happen that the decoding process will required more storage than
                                                               what was allocated (indexed by 'pos'). In this case it is better
                                                               to "kill" the individual (return 0). */

   for( int i = 0; i < r->quantity; ++i )
      if( r->symbols[i] >= TERMINAL_MIN )
      {
         phenotype[pos] = r->symbols[i];

#ifndef NOT_USING_T_CONST
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
#endif
         {
            if( r->symbols[i] >= ATTRIBUTE_MIN )
            {
               phenotype[pos] = T_ATTRIBUTE;
               ephemeral[pos] = r->symbols[i] - ATTRIBUTE_MIN;
            } 
         }
         ++pos;
      }
      else // It's a non-terminal, so calling recursively decode again...
      {
         pos = decode( genome, allele, phenotype, ephemeral, pos, r->symbols[i] );
         if( !pos ) return 0;
      }

   return pos;
}


/** ****************************************************************** **/
/** ************************* MAIN FUNCTIONS ************************* **/
/** ****************************************************************** **/

#include <interpreter_core_print>

void ppi_init( float** input, int nlin, int ncol, int argc, char** argv ) 
{
   data.argc = argc; data.argv = argv;
   CmdLine::Parser Opts( argc, argv );

   Opts.Bool.Add( "-v", "--verbose" );

   Opts.Bool.Add( "-machine" );

   Opts.Bool.Add( "-acc" );

   Opts.Int.Add( "-g", "--generations", 1000, 0, std::numeric_limits<int>::max() );

   Opts.Int.Add( "-s", "--seed", 0, 0, std::numeric_limits<long>::max() );

   Opts.Int.Add( "-ps", "--population-size", 1024, 1, std::numeric_limits<int>::max() );
   Opts.Int.Add( "-is", "--immigrants-size", 5, 1 );
   Opts.Float.Add( "-cp", "--crossover-probability", 0.90, 0.0, 1.0 );
   Opts.Float.Add( "-mr", "--mutation-rate", 0.01, 0.0, 1.0 );
   Opts.Int.Add( "-ts", "--tournament-size", 3, 1, std::numeric_limits<int>::max() );

   Opts.Bool.Add( "-e", "--elitism" );

   Opts.Int.Add( "-nb", "--number-of-bits", 2000, 16 );
   Opts.Int.Add( "-bg", "--bits-per-gene", 8, 8, 31 );
   Opts.Int.Add( "-bc", "--bits-per-constant", 16, 4, 63 );

   /* The --max-phenotype-size option (-mps) is useful when the computed
      theoretical maximum value of the phenotype may be too large, which leads
      to a large requirement of storage and stack size (during interpretation).
      Since it is very unlikely to hit the theoretical maximum phenotype size,
      in practical cases it is better to set a lower maximum value. This is
      the purpose of the -mps option. If -mps happens to be smaller than the
      ideal, the decoding process will simply invalidate the individual being
      decoded; i.e., there is no other side effect. */
   Opts.Int.Add( "-mps", "--max-phenotype-size", std::numeric_limits<int>::max(), 1 );

   Opts.Float.Add( "-min", "--min-constant", -10 );
   Opts.Float.Add( "-max", "--max-constant", 10 );

   Opts.String.Add( "-peers", "--address_of_island" );
 
   /* Maximum allowed number of generations without improvement [default = disabled] */
   Opts.Int.Add( "-st", "--stagnation-tolerance", numeric_limits<unsigned long>::max(), 0 );

   Opts.Float.Add( "-iat", "--immigrants-acceptance-threshold", 0.0, 0.0 );

   Opts.Int.Add( "-t", "--threads", -1, 0);

   // processing the command-line
   Opts.Process();

   // getting the results!
   data.verbose = Opts.Bool.Get("-v");

   data.machine = Opts.Bool.Get("-machine");

   data.generations = Opts.Int.Get("-g");

   data.seed = Opts.Int.Get("-s") == 0 ? time( NULL ) : Opts.Int.Get("-s");

   data.population_size = Opts.Int.Get("-ps");
   data.immigrants_size = Opts.Int.Get("-is");
   data.crossover_rate = Opts.Float.Get("-cp");
   data.mutation_rate = Opts.Float.Get("-mr");
   data.tournament_size = Opts.Int.Get("-ts");

   data.elitism = Opts.Bool.Get("-e");

   data.number_of_bits = Opts.Int.Get("-nb");

   // Parse bits_per_gene, but also increase it automatically if the given value (-bg) cannot hold the largest number of symbols of all rules
   data.bits_per_gene = Opts.Int.Get("-bg");
   const int number_of_rules = sizeof(gramatica)/sizeof(t_rule**);
   for ( int i = 0; i < number_of_rules; ++i )
   {
      int bits_needed = std::ceil( std::log10( tamanhos[i] ) / std::log10( 2.0 ) ); // log2 rounded up
      if ( bits_needed > data.bits_per_gene )
         data.bits_per_gene = bits_needed;
   }

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

   /*
      //////////////////////////////////////////////////////////////////////////
         Number of threads:
      //////////////////////////////////////////////////////////////////////////
      if '-t' is NOT given (i.e., -1), than uses only one thread (sequential)
      if '-t' is given as any value greater than 0, just set the number of threads to that value
      if '-t' is given as ZERO, than uses the maximum number of threads (= number of cores) or the value of the environment variable OMP_NUM_THREADS
   */
#ifdef _OPENMP
   if (Opts.Int.Get("-t") < 0) // Uses the default (threads = 1 = sequencial)
      omp_set_num_threads(1);
   else if (Opts.Int.Get("-t") > 0) // Uses the specified number of threads
      omp_set_num_threads(Opts.Int.Get("-t"));
#endif


   data.initial_symbol = NT_INICIAL;

   data.best_size = 1;
   data.best_individual.genome = NULL;
   data.best_individual.fitness = NULL;

   data.max_size_phenotype = std::min( MAX_QUANT_SIMBOLOS_POR_REGRA * data.number_of_bits/data.bits_per_gene, Opts.Int.Get<int>("-mps") );


   data.nlin = nlin;

   data.phenotype = new Symbol[data.population_size * data.max_size_phenotype];
   data.ephemeral = new float[data.population_size * data.max_size_phenotype];
   data.size = new int[data.population_size];
   data.sum_size = 0;

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

   data.parallel_version = Opts.Bool.Get("-acc");
   if( data.parallel_version )
   {
      if( acc_interpret_init( argc, argv, data.max_size_phenotype, MAX_QUANT_SIMBOLOS_POR_REGRA, data.population_size, input, nlin, ncol, 0, 0 ) )
      {
         fprintf(stderr,"Error in initialization phase.\n");
      }
   }
   else
   {
      seq_interpret_init( data.max_size_phenotype, input, nlin, ncol );
   }

   data.stagnation_tolerance = Opts.Int.Get( "-st" );
   double iat = Opts.Float.Get( "-iat" );
   if (iat < 1.0) // If in [0.0,1.0), then it is expressed as a percentage of '-st'
      Server::immigrants_acceptance_threshold = iat * data.stagnation_tolerance;
   else // if '>= 1.0', it is an absolute value
      Server::immigrants_acceptance_threshold = static_cast<long int>(iat);

   /* Storage for multi-threaded Random Number Generators (RNG), one for each OpenMP thread

      It is important to notice here that instead of allocating consecutive
      RNGs in a coalesced way, we do it by putting them in different memory
      regions (this is why we perform many separate allocations (new)). One
      should remember that each call to an RNG implies in updating the internal
      state variables. When running in OpenMP, if these state variables lied on
      the same cache line, then False Sharing would occur (when a thread writes
      to its state variables the cache line is invalidated), bringing down the
      parallel performance.
   */
   data.RNGs = new RNG*[GetMaxNumThreads()];
#pragma omp parallel for // Induces First Touch Policy, i.e., each RNG will be
                         // on the same core as the corresponding thread (hopefully)
   for (int i=0; i<GetMaxNumThreads(); ++i)
   {
      data.RNGs[i] = new RNG;
      data.RNGs[i]->Seed(data.seed ^ i);
   }

}

void ppi_clone( Population* original, int idx_original, Population* copy, int idx_copy )
{
#ifdef PROFILING
   util::Timer t_clone;
#endif

   const GENOME_TYPE* const org = original->genome[idx_original];
   GENOME_TYPE* cpy = copy->genome[idx_copy];

   for( int i = 0; i < data.number_of_bits; ++i ) cpy[i] = org[i];

   copy->fitness[idx_copy] = original->fitness[idx_original];

#ifdef PROFILING
   double elapsed = t_clone.elapsed();
#pragma omp atomic
   data.time_gen_clone   += elapsed;
#pragma omp atomic
   data.time_total_clone += elapsed;
#endif
}

int ppi_tournament( const float* fitness )
{
#ifdef PROFILING
   util::Timer t_tournament;
#endif

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

#ifdef PROFILING
   double elapsed = t_tournament.elapsed();
#pragma omp atomic
   data.time_gen_tournament   += elapsed;
#pragma omp atomic
   data.time_total_tournament += elapsed;
#endif

   return idx_winner;
}

void ppi_send_individual( Population* population )
{
#ifdef PROFILING
   util::Timer t_send;
#endif

   // Skip the first call, i.e, when it's the first generation (since it wasn't evaluated yet)
   static bool firstcall = true;
   if( firstcall ) {
      firstcall = false;
      return;
   }

   //std::cerr << data.peers.size() << std::endl;
   for( int i = 0; i < data.peers.size(); i++ )
   { 
      if( random_number() < data.peers[i].frequency )
      {
         //Testa se o thread ainda está mandando o indivíduo, escolhido na geração anterior, para a ilha.
         if( data.pool->isrunning[i] ) continue;

         const int idx = ppi_tournament( population->fitness );

         std::stringstream results;
         results <<  population->fitness[idx] << " ";
         for( int j = 0; j < data.number_of_bits; j++ )
            results <<  static_cast<int>(population->genome[idx][j]); /* NB: A cast to 'int' is necessary here otherwise it will send characters instead of digits when GENOME_TYPE == char */

         delete data.pool->clients[i]; delete data.pool->ss[i];

         data.pool->ss[i] = new StreamSocket();
         data.pool->clients[i] = new Client( *(data.pool->ss[i]), data.peers[i].address.c_str(), results.str(), data.pool->isrunning[i], data.pool->mutexes[i] );
         data.pool->isrunning[i] = 1;
         try {
            data.pool->threadpool.start( *(data.pool->clients[i]) );
         } catch (Poco::Exception& exc) {
            std::cerr << "ThreadPool.start(): " << exc.displayText() << std::endl;
         }

         if (data.verbose)
         {
#ifdef NDEBUG
            std::cerr << "^";
#else
            std::cerr << "\nSending Individual Thread[" << i << "] to " << data.peers[i].address << ": " << population->fitness[idx] << std::endl;
#endif
         }
      }
   }

#ifdef PROFILING
   data.time_total_send += t_send.elapsed();
#endif
}

int ppi_receive_individual( GENOME_TYPE** immigrants )
{
#ifdef PROFILING
   util::Timer t_receive;
#endif

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

      //std::cerr << "\nReceive::Receiving[slot=" << slot << "]: " << Server::m_immigrants[slot].data() << std::endl;
      if (data.verbose)
      {
#ifdef NDEBUG
         std::cerr << 'v';
#else
         std::cerr << "\nReceive::Receiving[slot=" << slot << "]: " << Server::m_fitness[slot] << std::endl;
#endif
      }

      /* It is possible that the number of char bits in 'tmp' is smaller than
       * 'number_of_bits', for instance when the parameter '-nb' of an external
       * island is smaller than the '-nb' of the current island. The line below
       * handles this case (it also takes into account the offset). */
      int chars_to_convert = std::min((int) data.number_of_bits, (int) Server::m_immigrants[slot].size() - offset - 1);
      //std::cerr << "\n[" << offset << ", " << Server::m_immigrants[slot].size() << ", " << chars_to_convert << ", " << data.number_of_bits << "]\n";
      for( int i = 0; i < chars_to_convert && tmp[i] != '\0'; i++ )
      {
         assert(tmp[i]-'0'==1 || tmp[i]-'0'==0); // In debug mode, assert that each value is either '0' or '1'
         immigrants[nImmigrants][i] = static_cast<bool>(tmp[i] - '0'); /* Ensures that the allele will be binary (0 or 1) regardless of the received value--this ensures it would work even if a communication error occurs (or a malicious message is sent). */
      }
      nImmigrants++;

      {
         Poco::FastMutex::ScopedLock lock( Server::m_mutex );

         Server::m_freeslots.push(slot);
      }

   }

#ifdef PROFILING
   data.time_gen_receive    = t_receive.elapsed();
   data.time_total_receive += t_receive.elapsed();
#endif

   return nImmigrants;
}

unsigned long ppi_evaluate( Population* descendentes, Population* antecedentes, int* nImmigrants )
{
#ifdef PROFILING
   unsigned long sum_size_gen = 0;
   util::Timer t_evaluate, t_decode;
   //int max_size = 0;
#endif

#ifdef PROFILING
#pragma omp parallel for reduction(+:sum_size_gen)
#else
#pragma omp parallel for
#endif
   for( int i = 0; i < data.population_size; i++ )
   {
      int allele = 0;
      data.size[i] = decode( descendentes->genome[i], &allele, data.phenotype + (i * data.max_size_phenotype), data.ephemeral + (i * data.max_size_phenotype), 0, data.initial_symbol );
#ifdef PROFILING
      sum_size_gen += data.size[i];
      //if( max_size < data.size[i] ) max_size = data.size[i];
#endif
   }
#ifdef PROFILING
   data.time_gen_decode     = t_decode.elapsed();
   data.time_total_decode  += t_decode.elapsed();
#endif

#ifdef PROFILING
   //std::cout << sum_size_gen/(double)data.population_size << " " << max_size << std::endl;
   data.sum_size += sum_size_gen;
#endif

   int index[data.best_size];

   if( data.parallel_version )
   {
      acc_interpret( data.phenotype, data.ephemeral, data.size, 
#ifdef PROFILING
      sum_size_gen, 
#endif
      descendentes->fitness, data.population_size, &ppi_send_individual, &ppi_receive_individual, antecedentes, nImmigrants, index, &data.best_size, 0, 0, ALPHA );
   }
   else
   {
      ////// Exchange (send and receive) individuals to and from islands.
      ////// TODO: Make them actually asynchronous via OpenMP.
      /* Send individuals to other islands. In order to allow for asynchronous
         send (which can be done in background while the population is
         evaluated), it will pick individuals from the last generation instead
         of the current one; this is a good trade-off, though.  Note that this
         function will skip the initial generation as it wasn't evaluated yet. */
      ppi_send_individual(antecedentes);

      /* Receive individuals from other islands. Individuals received from
       foreign islands will be put into the next population, which may sound
       strange, but it is the 'antecedentes' (in the swap operation,
       'antecedentes' will be made the next generation). */
      *nImmigrants = ppi_receive_individual( antecedentes->genome );

      seq_interpret( data.phenotype, data.ephemeral, data.size, 
#ifdef PROFILING
      sum_size_gen, 
#endif
      descendentes->fitness, data.population_size, index, &data.best_size, 0, 0, ALPHA );
   }

   for( int i = 0; i < data.best_size; i++ )
   {
      if( descendentes->fitness[index[i]] < data.best_individual.fitness[i] )
      {
         Server::stagnation = 0;
         ppi_clone( descendentes, index[i], &data.best_individual, i );
      }
      else
      {
         ++Server::stagnation;
      }
   }

#ifdef PROFILING
   data.time_gen_evaluate     = t_evaluate.elapsed();
   data.time_total_evaluate  += t_evaluate.elapsed();
   data.gpops_gen_evaluate    = (sum_size_gen * data.nlin) / t_evaluate.elapsed();
#endif

   return Server::stagnation;
}

void ppi_generate_population( Population* antecedentes, Population* descendentes, int* nImmigrants )
{
#ifdef PROFILING
   util::Timer t_generate;
#endif

#pragma omp parallel for
   for( int i = 0; i < data.population_size; ++i)
   {
      for( int j = 0; j < data.number_of_bits; j++ )
      {
         antecedentes->genome[i][j] = (random_number() < 0.5) ? 1 : 0;
      }
   }

#ifdef PROFILING
   data.time_generate = t_generate.elapsed();
#endif

   ppi_evaluate( antecedentes, descendentes, nImmigrants );
}

void ppi_crossover( const GENOME_TYPE* father, const GENOME_TYPE* mother, GENOME_TYPE* offspring1, GENOME_TYPE* offspring2 )
{
#ifdef PROFILING
   util::Timer t_crossover;
#endif

   if (GetRNG()->Probability(TWOPOINT_CROSSOVER_PROBABILITY))
   {
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
   } else {
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
   }
#ifdef PROFILING
   double elapsed = t_crossover.elapsed();
#pragma omp atomic
   data.time_gen_crossover   += elapsed;
#pragma omp atomic
   data.time_total_crossover += elapsed;
#endif
}

void ppi_mutation( GENOME_TYPE* genome )
{
#ifdef PROFILING
   util::Timer t_mutation;
#endif
   /* First the number of bit positions that will be mutated is chosen
    * (num_bits_mutated), based on 'mutation_rate', which defines the maximum
    * fraction of the vector of bits that can be mutated at once. */
   const int max_bits_mutated = ceil(data.mutation_rate * data.number_of_bits);
   int num_bits_mutated = (int) (random_number() * (max_bits_mutated+1));

   if (num_bits_mutated == 0) return; // lucky guy, no mutation for him...

   if (GetRNG()->Probability(BITFLIP_MUTATION_PROBABILITY))
   {
      //////////////////////////////////////////////////////////////////////////
      // Bit (allele) mutation
      //////////////////////////////////////////////////////////////////////////

      /* Then, each position is selected at random and its value is swapped. */
      while( num_bits_mutated-- > 0 )
      {
         int bit = (int)(random_number() * data.number_of_bits);
         genome[bit] = !genome[bit];
      }
   } else {
      //////////////////////////////////////////////////////////////////////////
      // Shrink mutation
      //////////////////////////////////////////////////////////////////////////

      /* An illustrative example of how this mutation works (it is very useful
         for simplifying solutions):

            Original genome:

                               ,> end
               [AAA|AAA|###|###|BBB|BBB|BBB]
                       L start


            Mutated:

                               ,> end
               [AAA|AAA|BBB|BBB|BBB|   |   ]
                       L start
       */

      /* This mutation works on entire genes, not single bits. Because of that,
         first we need to convert num_bits_mutated to a multiple of
         bits_per_gene (the next multiple of bits_per_gene). This equation
         below does exactly this. Note that the maximum number of bits to
         shrink is limited proportionally by 'mutation_rate'.
      */

      if (GetRNG()->Probability(AGGRESSIVE_SHRINK_MUTATION_PROBABILITY))
         num_bits_mutated = (int) (random_number() * (data.number_of_bits));

      int number_of_bits_to_shrink = int((num_bits_mutated+(data.bits_per_gene-1))/data.bits_per_gene)*data.bits_per_gene;

      /* Now we randomly choose the starting gene (again, this operates only on
         multiple of bits_per_gene): */
      int start = ((int)(random_number() * data.number_of_bits))/data.bits_per_gene * data.bits_per_gene;

      /* What if 'number_of_bits + gene_position' is greater than the size of
         the genome? We need to handle this situation: */
      int end = std::min(int(start+number_of_bits_to_shrink), int(data.number_of_bits));

      // Using memmove (overlapping) instead of a for-loop for efficiency
      memmove(&genome[start], &genome[end], (data.number_of_bits-end)*sizeof(GENOME_TYPE));
   }
#ifdef PROFILING
   double elapsed = t_mutation.elapsed();
#pragma omp atomic
   data.time_gen_mutation   += elapsed;
#pragma omp atomic
   data.time_total_mutation += elapsed;
#endif
}

void ppi_print_best( FILE* out, int generation, int print_mode ) 
{
   ppi_individual_print( &data.best_individual, 0, out, generation, data.argc, data.argv, print_mode );
}

#ifdef PROFILING
void ppi_print_time( bool total ) 
{

   double time_evolve     = total ? data.time_total_evolve : data.time_gen_evolve;
   double time_evaluate   = total ? data.time_total_evaluate : data.time_gen_evaluate;
   double time_decode     = total ? data.time_total_decode : data.time_gen_decode;
   double time_crossover  = total ? data.time_total_crossover : data.time_gen_crossover;
   double time_mutation   = total ? data.time_total_mutation : data.time_gen_mutation;
   double time_clone      = total ? data.time_total_clone : data.time_gen_clone;
   double time_tournament = total ? data.time_total_tournament : data.time_gen_tournament;
   double time_receive    = total ? data.time_total_receive : data.time_gen_receive;
   double gpops_evaluate  = total ? (data.sum_size * data.nlin) / data.time_total_evaluate : data.gpops_gen_evaluate;

   printf(", time_generate: %lf, time_decode: %lf, time_evolve: %lf, time_evaluate: %lf, time_crossover: %lf, time_mutation: %lf, time_clone: %lf, time_tournament: %lf, time_send: %lf, time_receive: %lf", data.time_generate, time_decode, time_evolve, time_evaluate, time_crossover, time_mutation, time_clone, time_tournament, data.time_total_send + Client::time, time_receive);
   if( data.parallel_version )
   {
      acc_print_time(total, data.sum_size);
   }
   else 
   {
      seq_print_time(total, data.sum_size);
   }
   printf(", gpops_evaluate: %lf\n", gpops_evaluate);
}
#endif

int ppi_evolve()
{
   /* Initialize the RNG seed */
   // FIXME (currently done in _init; should we put it here instead?) RNG::Seed(data.seed);

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

#ifdef PROFILING
   data.time_total_evaluate    = 0.0;
   data.time_total_crossover   = 0.0;
   data.time_total_mutation    = 0.0;
   data.time_total_clone       = 0.0;
   data.time_total_tournament  = 0.0;
   data.time_total_send        = 0.0;
   data.time_total_receive     = 0.0;
#endif
   
   Population antecedentes, descendentes;

   antecedentes.fitness = new float[data.population_size];
   descendentes.fitness = new float[data.population_size];

   antecedentes.genome = new GENOME_TYPE*[data.population_size];
   descendentes.genome = new GENOME_TYPE*[data.population_size];
#pragma omp parallel for
   for (int i=0; i < data.population_size; ++i)
   {
      antecedentes.genome[i] = new GENOME_TYPE[data.number_of_bits];
      descendentes.genome[i] = new GENOME_TYPE[data.number_of_bits];
   }


   data.best_individual.fitness = new float[data.best_size];
   data.best_individual.genome = new GENOME_TYPE*[data.best_size];
   for( int i = 0; i < data.best_size; i++ )
   {
      data.best_individual.fitness[i] = std::numeric_limits<float>::max();
      data.best_individual.genome[i] = new GENOME_TYPE[data.number_of_bits];
   }

   int nImmigrants;

#ifdef PROFILING
   util::Timer t_gen_evolve;
#endif

   // 1 e 2:
   //cerr << "\nGeneration[0]  ";
   ppi_generate_population( &antecedentes, &descendentes, &nImmigrants );

#ifdef PROFILING
      data.time_total_evolve = t_gen_evolve.elapsed();
#endif

   // 3:
   int geracao;
   for( geracao = 1; geracao <= data.generations; ++geracao )
   {
#ifdef PROFILING
      t_gen_evolve.restart();

      data.time_gen_crossover  = 0.0;
      data.time_gen_mutation   = 0.0;
      data.time_gen_clone      = 0.0;
      data.time_gen_tournament = 0.0;
#endif

      // 4:
      if( data.elitism ) 
      {
         ppi_clone( &data.best_individual, 0, &descendentes, nImmigrants );
         nImmigrants++;
      }

      //std::cerr << "\nnImmigrants[generation: " << geracao << "]: " << nImmigrants << std::endl;

      // 5
#pragma omp parallel for
      for( int i = nImmigrants; i < data.population_size; i += 2 )
      {
         // 6:
         int idx_father = ppi_tournament( antecedentes.fitness );
         int idx_mother = ppi_tournament( antecedentes.fitness );

         // 7:
         if( random_number() < data.crossover_rate )
         {
            // 8 e 9:
            if( i < ( data.population_size - 1 ) )
            {
               ppi_crossover( antecedentes.genome[idx_father], antecedentes.genome[idx_mother], descendentes.genome[i], descendentes.genome[i + 1] );
            }
            else 
            {
               ppi_crossover( antecedentes.genome[idx_father], antecedentes.genome[idx_mother], descendentes.genome[i], descendentes.genome[i] );
            }
         } // 10
         else 
         {
            // 9:
            ppi_clone( &antecedentes, idx_father, &descendentes, i );
            if( i < ( data.population_size - 1 ) )
            {
               ppi_clone( &antecedentes, idx_mother, &descendentes, i + 1 );
            }
         } // 10

         // 11, 12, 13, 14 e 15:
         ppi_mutation( descendentes.genome[i] );
         if( i < ( data.population_size - 1 ) )
         {
            ppi_mutation( descendentes.genome[i + 1] );
         }
      } // 16

      // 17:
      if (ppi_evaluate( &descendentes, &antecedentes, &nImmigrants ) > data.stagnation_tolerance) geracao = data.generations;

      // 18:
      swap( &antecedentes, &descendentes );

#ifdef PROFILING
      data.time_gen_evolve    = t_gen_evolve.elapsed();
      data.time_total_evolve += t_gen_evolve.elapsed();
#endif

      if (data.verbose)
      {
         if (Server::stagnation == 0 || geracao < 2) {
            if (data.machine) { // Output meant to be consumed by scripts, not humans
               ppi_print_best(stdout, geracao, 1);
#ifdef PROFILING
               ppi_print_time(false);
#else
               fprintf(stdout, "\n");
#endif
            } else
               ppi_print_best(stdout, geracao, 0);

         }
         else std::cerr << '.';
      }
   } // 19


   // Clean up
   for (int i=0; i < data.population_size; ++i)
   {
      delete[] antecedentes.genome[i];
      delete[] descendentes.genome[i];
   }
   delete[] antecedentes.genome;
   delete[] descendentes.genome;
   delete[] antecedentes.fitness;
   delete[] descendentes.fitness;

   return geracao;
}

void ppi_destroy()
{
   // Wait for awhile then kill the non-finished threads. This ensures a
   // graceful termination and prevents segmentation faults at the end
   Poco::ThreadPool::defaultPool().stopAll();
   data.pool->threadpool.stopAll();

   for( int i = 0; i < data.best_size; i++ )
   {
      delete[] data.best_individual.genome[i];
   }
   delete[] data.best_individual.genome;
   delete[] data.best_individual.fitness;
   delete[] data.phenotype;
   delete[] data.ephemeral;
   delete[] data.size;
   delete[] Server::m_immigrants;
   delete[] Server::m_fitness;
   for (int i=0; i<GetMaxNumThreads(); ++i) delete data.RNGs[i]; delete[] data.RNGs;

   delete data.pool;

   if( !data.parallel_version ) {seq_interpret_destroy();}
}

}
