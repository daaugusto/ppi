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

#ifndef ppi_h
#define ppi_h


#include "client/client.h"
#include "Poco/ThreadPool.h"

namespace ppi {

/** Funcoes exportadas **/
/** ****************************************************************************************** **/
/** ************************************* Function init ************************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void ppi_init( float** input, int nlin, int ncol, int argc, char** argv );

/** ****************************************************************************************** **/
/** ************************************* Function evolve ************************************ **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
int ppi_evolve();

/** ****************************************************************************************** **/
/** ********************************** Function print_best *********************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void ppi_print_best( FILE* out, int generation, int print_mode );

/** ****************************************************************************************** **/
/** ********************************** Function print_time *********************************** **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
#ifdef PROFILING
void ppi_print_time( bool total=false );
#endif

/** ****************************************************************************************** **/
/** ************************************ Function destroy ************************************ **/
/** ****************************************************************************************** **/
/**                                                                                            **/
/** ****************************************************************************************** **/
void ppi_destroy();

/******************************************************************************/
class Pool {
public:

   Pool(unsigned size): ss(size, NULL), clients(size, NULL)
   {
      /* Expands threadpool if the available number of threads is less than
       * the number of peers, otherwise the exception "No thread available" is
       * thrown. */
      if (threadpool.available() < size)
         threadpool.addCapacity(size - (threadpool.available()));

      // Create an array of mutexes (one for each id)
      mutexes = new Poco::FastMutex[size];

      // Create and initialize an array of isrunning's (one for each id)
      isrunning = new bool[size];
      for (unsigned i = 0; i < size; i++) isrunning[i] = false;
   }

   ~Pool()
   {
      // Clean up!

      delete[] isrunning;
      delete[] mutexes;

      for (unsigned i = 0; i < clients.size(); i++)
      {
         delete clients[i];
         delete ss[i];
      }
   }

   std::vector<StreamSocket*> ss;
   std::vector<Client*> clients;

   bool * isrunning;
   Poco::ThreadPool threadpool;
   Poco::FastMutex * mutexes;
};
/******************************************************************************/

}

#endif
