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

#include <stdlib.h>
#include <stdio.h> 
#include <cmath>    
#include <fstream>
#include "server/server.h"
#include "Poco/Exception.h"
#include "util/CmdLineParser.h"
#include "util/Exception.h"
#include "util/Util.h"
#include "ppi.h"
#include "ppp.h"

/** ****************************************************************** **/
/** *********************** AUXILIARY FUNCTIONS ********************** **/
/** ****************************************************************** **/

int read( const std::string& dataset, float**& input, int &ncol, int& nlin )
{
   std::ifstream infile( dataset.c_str() );

   if (!infile) {
      fprintf(stderr, "Failed to open dataset file '%s' (use '-d dataset', where dataset is the path of the training CSV file)\n", dataset.c_str());
      return 2;
   }

   std::string line; std::string token;

   ncol = 0; nlin = 0; float tmp;
   bool header = true;
   while( std::getline(infile, line) )
   {
      if( !line.empty() && line[0] != '#' ) 
      { 
         std::istringstream iss( line );

         if( header )
         {
            while( std::getline(iss, token, ',') ) { ncol++; }
            if ( util::StringTo<float>(tmp, token) ) { header = false; }
         }

         if( header ) { header = false; } 
         else { nlin++; }
      }
   }

   input = new float*[nlin];
   for( int i = 0; i < nlin; i++ )
     input[i] = new float[ncol];

   infile.clear();
   infile.seekg(0);

   int i = 0; int k = 0;
   header = true;
   while( std::getline(infile, line) ) 
   {
      if( !line.empty() && line[0] != '#' ) 
      {

         std::istringstream iss( line );

         int j = 0;
         while( std::getline(iss, token, ',') )
         {
            if( header )
            {
               if ( !util::StringTo<float>(tmp, token) ) { break; }
               else { header = false; }
            }

            if ( !util::StringTo<float>(input[i][j], token) ) 
            {
               if( header )
                  header =  false;
               else 
               {
                  fprintf(stderr, "Invalid input at line %d, column %d.\n", k+1, j+1);
                  return 2;
               }
            }
            if( std::isnan( input[i][j]) || std::isinf(input[i][j]) )
            {
               fprintf(stderr, "Invalid input at line %d, column %d.\n", k+1, j+1);
               return 2;
            }
            //std::cout << input[i][j] << " ";
            j++;
         }
         if( !header )
         {
            if( ncol != j )
            {
               fprintf(stderr,"Line '%d' has '%d' columns but the expected number is '%d'.\n", k+1, j, ncol);
               return 1;
            }
         }
         if( header ) { header = false; }
         else { i++; ncol = j; }//std::cout << std::endl; }
      }
      k++;
   }
   return 0;

   //if( scanf(token.c_str(),"%f%*[^\n],",&input[i][j]) != 1 || isnan(input[i][j]) || isinf(input[i][j]) )
   //if( scanf(token.c_str(),"%f,",&input[i][j]) != 1 || isnan(input[i][j]) || isinf(input[i][j]) )
}

void destroy( float** input, int nlin )
{
   for( int i = 0; i < nlin; ++i )
     delete [] input[i];
   delete [] input;
}


int main(int argc, char** argv)
{
#ifdef PROFILING
   util::Timer t_total;
#endif
   try {
      CmdLine::Parser Opts( argc, argv );

      Opts.Int.Add( "-port", "--number_of_port" );
      Opts.String.Add( "-d", "--dataset" );
      Opts.String.Add( "-sol", "--solution" );

      Opts.Process();

      Common::SetupLogger( "information" );

      int nlin; int ncol;
      float** input;


      int error = read( Opts.String.Get("-d"), input, ncol, nlin );
      if ( error ) {return error;}

      if( Opts.String.Found("-sol") )
      {
         ppi::ppp_init( input, nlin, ncol, argc, argv );
         ppi::ppp_interpret();
         ppi::ppp_print( stdout );
         ppi::ppp_destroy();
      }
      else
      {
         ServerSocket svs(SocketAddress("0.0.0.0", Opts.Int.Get("-port")));
         TCPServerParams* pParams = new TCPServerParams;
         //pParams->setMaxThreads(4);
         //pParams->setMaxQueued(4);
         //pParams->setThreadIdleTime(1000);
         TCPServer srv(new TCPServerConnectionFactoryImpl<Server>(), svs, pParams);
         srv.start();
         //sleep(100);

         ppi::ppi_init( input, nlin, ncol, argc, argv );
         int generations = ppi::ppi_evolve();

         fprintf(stdout, "\n> Overall best:");
         ppi::ppi_print_best( stdout, generations, 1 );
#ifdef PROFILING
         printf(";time_total: %lf", t_total.elapsed());
         ppi::ppi_print_time(false);
#else
         printf("\n");
#endif
         ppi::ppi_destroy();
      }

      destroy(input, nlin);
   }
   catch( const CmdLine::E_Exception& e ) {
      std::cerr << e;
      return 1;
   }
   catch( const Exception& e ) {
      std::cerr << e;
      return 2;
   }
   catch( const Poco::Exception& e ) {
      std::cerr << '\n' << "> Error: " << e.displayText() << " [@ Poco]\n";
      return 4;
   }
   catch( const std::exception& e ) {
      std::cerr << '\n' << "> Error: " << e.what() << std::endl;
      return 8;
   }
   catch( ... ) {
      std::cerr << '\n' << "> Error: " << "An unknown error occurred." << std::endl;
      return 16;
   }

   return 0;
}
