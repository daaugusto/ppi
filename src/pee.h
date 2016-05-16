/** TAD: PEE **/

#ifndef pee_h
#define pee_h

#include "client/client.h"

/** Funcoes exportadas **/
/** ****************************************************************************************** **/
/** ************************************* Function init ************************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void pee_init( float** input, int nlin, int argc, char** argv );

/** ****************************************************************************************** **/
/** ************************************* Function evolve ************************************ **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void pee_evolve();

/** ****************************************************************************************** **/
/** ********************************** Function print_best *********************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void pee_print_best( FILE* out, int print_mode );

/** ****************************************************************************************** **/
/** ********************************** Function print_time *********************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void pee_print_time();

/** ****************************************************************************************** **/
/** ************************************ Function destroy ************************************ **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void pee_destroy();

/******************************************************************************/
class Pool {
public:

   Pool( unsigned size ): threads( size ), ss( size, NULL), clients( size, NULL ), starts( size ) 
   {
      for( int i = 0; i < threads.size(); i++ )
      {
         threads[i] = new Poco::Thread();
         if( threads[i]->isRunning() ) { std::cerr << "Is running..." << std::endl; }
         starts[i] = false;
      }
   }
 
   ~Pool()
   {
      for( unsigned i = 0; i < clients.size(); i++ )
      {
         //Testa se o thread ainda está mandando o indivíduo para a ilha. Caso afirmativo, 
         //it will wait until the thread terminates.
         if( threads[i]->isRunning() ) { threads[i]->join(); }
         //Clean up. They were allocated in pee_send_individual (pee.cc)
         delete threads[i], clients[i], ss[i];
      } 
   }

   std::vector<Poco::Thread*> threads;
   std::vector<StreamSocket*> ss;
   std::vector<Client*> clients;
   std::vector<bool> starts;
};
/******************************************************************************/

#endif

