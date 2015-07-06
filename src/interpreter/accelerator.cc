// ------------------------------------------------------------------------
// Habilita disparar exceções C++
#define __CL_ENABLE_EXCEPTIONS

// Cabeçalho OpenCL para C++
#include <CL/cl.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#include <limits>
#include <string>   
#include <vector>
#include <utility>
#include <iostream> 
#include <fstream> 
#include "accelerator.h"
#include "../util/CmdLineParser.h"

using namespace std;

/** ****************************************************************** **/
/** ***************************** TYPES ****************************** **/
/** ****************************************************************** **/

static struct t_data { int max_size; int nlin; unsigned local_size; unsigned global_size; std::string strategy; vector<cl::Device> device; cl::Context context; cl::Kernel kernel; cl::CommandQueue queue; cl::Buffer buffer_phenotype; cl::Buffer buffer_ephemeral; cl::Buffer buffer_size; cl::Buffer buffer_inputs; cl::Buffer buffer_vector; } data;


/** ****************************************************************** **/
/** *********************** AUXILIARY FUNCTION *********************** **/
/** ****************************************************************** **/

// -----------------------------------------------------------------------------
int opencl_init( int platform_id, int device_id, int type )
{
   vector<cl::Platform> platforms;

   /* Iterate over the available platforms and pick the list of compatible devices
      from the first platform that offers the device type we are querying. */
   cl::Platform::get( &platforms );

   vector<cl::Device> devices;

   /*

   Possible options:

        type  device  platform

   1.   -1      X        X
   2.   -1     -1        X     --> device = 0
   3.   -1      X       -1     --> platform = 0
   4.    X     -1        X     
   5.    X     -1       -1 
   6.   -1     -1       -1     --> platform = last_platform; device = 0 

   */

   if( platform_id < 0 && device_id >= 0 ) // option 3
   {
      platform_id = 0; 
   }

   // Check if the user gave us a valid platform 
   if( platform_id >= (int) platforms.size() )
   {
      fprintf(stderr, "Valid platform range: [0, %d].\n", (int) (platforms.size()-1));
      return 1;
   }

   bool leave = false;

   int first_platform = platform_id >= 0 ? platform_id : 0;
   int last_platform  = platform_id >= 0 ? platform_id + 1 : platforms.size();
   for( int m = first_platform; m < last_platform; m++ )
   {
      platforms[m].getDevices( CL_DEVICE_TYPE_ALL, &devices );

      // Check if the user gave us a valid device 
      if( device_id >= (int) devices.size() ) 
      {
         fprintf(stderr, "Valid device range: [0, %d].\n", (int) (devices.size()-1));
         return 1;
      }

      int first_device = device_id >= 0 ? device_id : 0;
      data.device[0] = devices[first_device];

      if( type >= 0 && device_id < 0 ) // options 4 e 5
      {
         for ( int n = 0; n < devices.size(); n++ )
         {
            if ( devices[n].getInfo<CL_DEVICE_TYPE>() == type ) 
            {
               leave = true;
               data.device[0] = devices[n];
               break;
            }
         }
         if( leave ) break;
      }
   }

   if( type >= 0 && !leave )
   {
      fprintf(stderr, "Not a single compatible device found.\n");
      return 1;
   }

   data.context = cl::Context( data.device );

   data.queue = cl::CommandQueue( data.context, data.device[0], CL_QUEUE_PROFILING_ENABLE );

   return 0;
}

// -----------------------------------------------------------------------------
int build_kernel( int population_size )
{
   ifstream file("accelerator.cl");
   string kernel_str( istreambuf_iterator<char>(file), ( istreambuf_iterator<char>()) );

   cl::Program::Sources source( 1, make_pair( kernel_str.c_str(), kernel_str.size() ) );

   cl::Program program( data.context, source );

   char buildOptions[60];
   sprintf( buildOptions, "-DMAX_PHENOTYPE_SIZE=%u -I.", data.max_size );  
   try {
      program.build( data.device, buildOptions );
   }
   catch( cl::Error& e )
   {
      cerr << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(data.device[0]) << std::endl;
      throw;
   }

   unsigned max_cu = data.device[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
   unsigned max_local_size = fmin( data.device[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), data.device[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] );

   if( data.strategy == "PP" ) 
   {
      data.local_size = 1;
      data.global_size = population_size;
      data.kernel = cl::Kernel( program, "evaluate_pp" );
   }
   else
   {
      if( data.strategy == "FP" ) 
      {
         // Evenly distribute the workload among the compute units (but avoiding local size
         // being more than the maximum allowed).
         data.local_size = fmin( max_local_size, (unsigned) ceil( data.nlin/(float) max_cu ) );

         // It is better to have global size divisible by local size
         data.global_size = (unsigned) ( ceil( data.nlin/(float) data.local_size ) * data.local_size );
         data.kernel = cl::Kernel( program, "evaluate_fp" );
      }
      else
      {
         if( data.strategy == "PPCU" ) 
         {
            if( data.nlin < max_local_size )
            {
               data.local_size = data.nlin;
            }
            else
            {
               data.local_size = max_local_size;
            }
            // One individual per work-group
            data.global_size = population_size * data.local_size;
            data.kernel = cl::Kernel( program, "evaluate_ppcu" );
         }
         else
         {
            fprintf(stderr, "Not a compatible strategy found.\n");
            return 1;
         }
      }
   }

   return 0;
}

// -----------------------------------------------------------------------------
void create_buffers( const unsigned population_size, float** input, float** model, float* obs, int ninput, int nmodel, int prediction_mode )
{
   int ncol = ninput + nmodel + 1;

   // Buffer (memory on the device) of training points (input, model and obs)
   data.buffer_inputs = cl::Buffer( data.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, data.nlin * ncol * sizeof( float ) );

   float* inputs = (float*) data.queue.enqueueMapBuffer( data.buffer_inputs, CL_TRUE, CL_MAP_WRITE, 0, data.nlin * ncol * sizeof( float ) );

   if( data.strategy == "PP" ) 
   {
      for( int i = 0; i < data.nlin; i++ )
      {
         for( int j = 0; j < ninput; j++ )
         {
            inputs[i * ncol + j] = input[i][j];
         }
         for( int j = 0; j < nmodel; j++ )
         {
            inputs[i * ncol + (j + ninput)] = model[i][j];
         }
         inputs[i * ncol + (nmodel + ninput)] = obs[i];
      }
   }
   else
   {
      // Transposed version for coalesced access on the GPU
      /*
      TRANSPOSITION (for coalesced access on the GPU)

                                 Transformed and linearized data points
                                 +----------++----------+   +----------+
                                 | 1 2     q|| 1 2     q|   | 1 2     q|
           +-------------------> |X X ... X ||X X ... X |...|X X ... X |
           |                     | 1 1     1|| 2 2     2|   | p p     p|
           |                     +----------++----------+   +----------+
           |                                ^             ^
           |    ____________________________|             |
           |   |       ___________________________________|
           |   |      |
         +--++--+   +--+
         | 1|| 1|   | 1|
         |X ||X |...|X |
         | 1|| 2|   | p|
         |  ||  |   |  |
         | 2|| 2|   | 2|
         |X ||X |...|X |
         | 1|| 2|   | p|
         |. ||. |   |. |
         |. ||. |   |. |
         |. ||. |   |. |
         | q|| q|   | q|
         |X ||X |...|X |
         | 1|| 2|   | p|
         +--++--+   +--+
      Original data points

      */
      if( data.strategy == "FP" || data.strategy == "PPCU" ) 
      {
         for( int i = 0; i < data.nlin; i++ )
         {
            for( int j = 0; j < ninput; j++ )
            {
               inputs[j * data.nlin + i] = input[i][j];
            }
            for( int j = 0; j < nmodel; j++ )
            {
               inputs[(j + ninput) * data.nlin + i] = model[i][j];
            }
            inputs[(nmodel + ninput) * data.nlin + i] = obs[i];
         }
      }
      else
      {
         fprintf(stderr, "Not a compatible strategy found.\n");
      }
   }
   // Unmapping
   data.queue.enqueueUnmapMemObject( data.buffer_inputs, inputs ); 

   //inputs = (float*) data.queue.enqueueMapBuffer( data.buffer_inputs, CL_TRUE, CL_MAP_READ, 0, data.nlin * ncol * sizeof( float ) );
   //for( int i = 0; i < data.nlin * ncol; i++ )
   //{
   //   if( i % data.nlin == 0 ) fprintf(stdout,"\n");
   //   fprintf(stdout,"%f ", inputs[i]);
   //}
   //data.queue.enqueueUnmapMemObject( data.buffer_inputs, inputs ); 

   // Buffer (memory on the device) of the programs
   data.buffer_phenotype = cl::Buffer( data.context, CL_MEM_READ_ONLY, data.max_size * population_size * sizeof( Symbol ) );
   data.buffer_ephemeral = cl::Buffer( data.context, CL_MEM_READ_ONLY, data.max_size * population_size * sizeof( float ) );
   data.buffer_size      = cl::Buffer( data.context, CL_MEM_READ_ONLY, population_size * sizeof( int ) );

   if( prediction_mode ) // Buffer (memory on the device) of prediction (one por example)
   { 
      data.buffer_vector = cl::Buffer( data.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, data.nlin * sizeof( float ) );
   }
   else // Buffer (memory on the device) of prediction errors
   {
      if( data.strategy == "FP" ) 
      {
         data.buffer_vector = cl::Buffer( data.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, (data.global_size/data.local_size) * population_size * sizeof( float ) );
      }
      else
      {
         if( data.strategy == "PPCU" || data.strategy == "PP" ) // (one por program)
         {
            data.buffer_vector = cl::Buffer( data.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, population_size * sizeof( float ) );
         }
      }
   }

   //std::cout << "\nDevice: " << data.device[0].getInfo<CL_DEVICE_NAME>() << ", Compute units: " << max_cu << ", Max local size: " << max_local_size << std::endl;
   //std::cout << "\nLocal size: " << data.local_size << ", Global size: " << data.global_size << ", Work groups: " << data.global_size/data.local_size << "\n" << std::endl;

   data.kernel.setArg( 0, data.buffer_phenotype );
   data.kernel.setArg( 1, data.buffer_ephemeral );
   data.kernel.setArg( 2, data.buffer_size );
   data.kernel.setArg( 3, data.buffer_inputs );
   data.kernel.setArg( 4, data.buffer_vector );
   data.kernel.setArg( 5, sizeof( float ) * data.local_size, NULL );
   data.kernel.setArg( 6, data.nlin );
   data.kernel.setArg( 7, ncol );
   data.kernel.setArg( 8, prediction_mode );
}


/** ****************************************************************** **/
/** ************************* MAIN FUNCTION ************************** **/
/** ****************************************************************** **/

// -----------------------------------------------------------------------------
int acc_interpret_init( int argc, char** argv, const unsigned size, const unsigned population_size, float** input, float** model, float* obs, int nlin, int prediction_mode )
{
   CmdLine::Parser Opts( argc, argv );

   Opts.Int.Add( "-ni", "--number_of_inputs" );
   Opts.Int.Add( "-nm", "--number_of_models" );
   Opts.Int.Add( "-platform-id", "", -1, 0 );
   Opts.Int.Add( "-device-id", "", -1, 0 );
   Opts.String.Add( "-type" );
   Opts.String.Add( "-strategy" );
   Opts.Process();
   data.strategy = Opts.String.Get("-strategy");
   data.max_size = size;
   data.nlin = nlin;

   int type = -1;
   if( Opts.String.Found("-type") )
   {
      if( !strcmp(Opts.String.Get("-type").c_str(),"CPU") ) 
      { 
         type = CL_DEVICE_TYPE_CPU; 
      }
      else 
      {
         if( !strcmp(Opts.String.Get("-type").c_str(),"GPU") ) 
         {
            type = CL_DEVICE_TYPE_GPU; 
         }
         else
         {
            fprintf(stderr, "Not a single compatible device found.\n");
            return 1;
         }
      }
   }

   if ( opencl_init( Opts.Int.Get("-platform-id"), Opts.Int.Get("-device-id"), type ) )
   {
      fprintf(stderr,"Error in OpenCL initialization phase.\n");
      return 1;
   }

   if ( build_kernel( population_size ) )
   {
      fprintf(stderr,"Error in build the kernel.\n");
      return 1;
   }

   create_buffers( population_size, input, model, obs, Opts.Int.Get("-ni"), Opts.Int.Get("-nm"), prediction_mode );

//   try
//   {
//   }
//   catch( cl::Error& e )
//   {
//      cerr << "ERROR: " << e.what() << " ( " << e.err() << " )\n";
//   }

   return 0;
}

// -----------------------------------------------------------------------------
void acc_interpret( Symbol* phenotype, float* ephemeral, int* size, float* vector, int nInd, int prediction_mode )
{
   data.queue.enqueueWriteBuffer( data.buffer_phenotype, CL_TRUE, 0, data.max_size * nInd * sizeof( Symbol ), phenotype );
   data.queue.enqueueWriteBuffer( data.buffer_ephemeral, CL_TRUE, 0, data.max_size * nInd * sizeof( float ), ephemeral ); 
   data.queue.enqueueWriteBuffer( data.buffer_size, CL_TRUE, 0, nInd * sizeof( int ), size ); 

   if( data.strategy == "FP" ) 
   {
      data.kernel.setArg( 9, nInd );
   }

   // ---------- begin kernel execution
   data.queue.enqueueNDRangeKernel( data.kernel, cl::NDRange(), cl::NDRange( data.global_size ), cl::NDRange( data.local_size ) );
   // ---------- end kernel execution

   // Wait until the kernel has finished
   data.queue.finish();

   float *tmp;
   if ( prediction_mode )
   {
      tmp = (float*) data.queue.enqueueMapBuffer( data.buffer_vector, CL_TRUE, CL_MAP_READ, 0, data.nlin * sizeof( float ) );
      for( int i = 0; i < data.nlin; i++ ) {vector[i] = tmp[i];}
   }
   else
   {
      if( data.strategy == "FP" ) 
      {
         // -----------------------------------------------------------------------
         /*
            Each kernel execution will put in data.buffer_vector the partial errors of each individual:

            |  0 |  0 |     |  0    ||   1 |  1 |     |  1   |     |  ind-1 |  ind-1 |     |  ind-1 |
            | E  | E  | ... | E     ||  E  | E  | ... | E    | ... | E      | E      | ... | E      |        
            |  0 |  1 |     |  n-1  ||   0 |  1 |     |  n-1 |     |  0     |  1     |     |  n-1   |    

            where 'ind-1' is the index of the last individual, and 'n-1' is the index of the
            last 'partial error', that is, 'n-1' is the index of the last work-group (gr_id).
          */

         const unsigned num_work_groups = data.global_size / data.local_size;

         // The line below maps the contents of 'data_buffer_vector' into 'tmp'.
         tmp = (float*) data.queue.enqueueMapBuffer( data.buffer_vector, CL_TRUE, CL_MAP_READ, 0, num_work_groups * nInd * sizeof( float ) );
         
         float sum;

         // Reduction on host!
         for( int ind = 0; ind < nInd; ind++)
         {
            sum = 0.0;
            for( int gr_id = 0; gr_id < num_work_groups; gr_id++ ) 
               sum += tmp[ind * num_work_groups + gr_id];

            if( isnan( sum ) || isinf( sum ) ) 
               vector[ind] = std::numeric_limits<float>::max();
            else 
               vector[ind] = sum/data.nlin;
         }
      }
      else
      {
         if( data.strategy == "PPCU" || data.strategy == "PP" ) 
         {
            tmp = (float*) data.queue.enqueueMapBuffer( data.buffer_vector, CL_TRUE, CL_MAP_READ, 0, nInd * sizeof( float ) );
            for( int i = 0; i < nInd; i++ ) {vector[i] = tmp[i];}
         }
      }
   }
   data.queue.enqueueUnmapMemObject( data.buffer_vector, tmp ); 
}
