#include <stdlib.h>
#include <stdio.h> 
#include <cmath>    
#include "util/CmdLineParser.h"
#include "util/Exception.h"
#include "pee.h"
#include "pep.h"
#include "server/server.h"
#include "Poco/Exception.h"

//TODO fazer comentários; passar para inglês

/** ****************************************************************** **/
/** *********************** AUXILIARY FUNCTIONS ********************** **/
/** ****************************************************************** **/

int read( const std::string& dataset, float**& input, int ncol, int& nlin )
{
   char c;
   FILE *arqentra;

   arqentra = fopen(dataset.c_str(),"r");
   if(arqentra == NULL) {
     fprintf(stderr, "Could not open file for reading (%s).\n", dataset.c_str());
     return 1;
   }
 
   nlin = 0;
   while( (c = fgetc(arqentra)) != EOF )
       if ( c == '\n' ) {nlin++;}

   input   = new float*[nlin];
   for( int i = 0; i < nlin; i++ )
     input[i] = new float[ncol];

   rewind(arqentra);
   for( int i = 0; i < nlin; i++ )
   {
      for( int j = 0; j < ncol; j++ )
      {
         if( j == (ncol-1) )
         {
            if( fscanf(arqentra,"%f%*[^\n],",&input[i][j]) != 1 || isnan(input[i][j]) || isinf(input[i][j]) )
            {
               fprintf(stderr, "Invalid input at line %d and column %d.\n", i, j);
               return 2;
            }
         }
         else
         {
            if( fscanf(arqentra,"%f,",&input[i][j]) != 1 || isnan(input[i][j]) || isinf(input[i][j]) )
            {
               fprintf(stderr, "Invalid input at line %d and column %d.\n", i, j);
               return 2;
            }
         }
      }
   }
   fclose (arqentra);
   return 0;
}

void destroy( float** input, int nlin )
{
   for( int i = 0; i < nlin; ++i )
     delete [] input[i];
   delete [] input;
}


int main(int argc, char** argv)
{
   try {
      CmdLine::Parser Opts( argc, argv );

      Opts.Int.Add( "-ncol", "--number_of_columns" );
      Opts.Int.Add( "-port", "--number_of_port" );
      Opts.String.Add( "-d", "--dataset" );
      Opts.String.Add( "-sol", "--solution_file" );

      Opts.Process();

      Common::SetupLogger( "information" );

      int ncol = Opts.Int.Get("-ncol");   

      int nlin;
      float** input;

      int error = read( Opts.String.Get("-d"), input, ncol, nlin );
      if ( error ) {return error;}

      if( Opts.String.Found("-sol") )
      {
         pep_init( input, nlin, argc, argv );
         pep_interpret();
         pep_print( stdout );
         pep_destroy();
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

         pee_init( input, nlin, argc, argv );
         int generations = pee_evolve();
         fprintf(stdout, "\n> Overall best:");
         pee_print_best( stdout, generations, 1 );
         pee_print_time();
         pee_destroy();
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
