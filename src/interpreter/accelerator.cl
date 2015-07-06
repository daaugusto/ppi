#include "symbol"

__kernel void
evaluate_pp( __global const Symbol* phenotype, __global const float* ephemeral, __global const int* size, __global const float* inputs, __global float* vector, __local float* PE, int nlin, int ncol, int prediction_mode )
{
   float stack[MAX_PHENOTYPE_SIZE];
   int stack_top;

   int gl_id = get_global_id(0);

   if( size[gl_id] == 0 && !prediction_mode )
   {
      vector[gl_id] = MAXFLOAT;
   }
   else
   {
      PE[0] = 0.0f;
      for( int n = 0; n < nlin; ++n )
      {
         stack_top = -1;
         for( int i = size[gl_id] - 1; i >= 0; --i )
         {
            switch( phenotype[gl_id * MAX_PHENOTYPE_SIZE + i] )
            {
               #include "core"
               case T_ATTRIBUTE:
                  stack[++stack_top] = inputs[n * ncol + (int)ephemeral[gl_id * MAX_PHENOTYPE_SIZE + i]];
                  break;
               case T_CONST:
                  stack[++stack_top] = ephemeral[gl_id * MAX_PHENOTYPE_SIZE + i];
                  break;
               default:
                  break;
            }
         }
         if( !prediction_mode )
         {
            PE[0] += fabs( stack[stack_top] - inputs[n * ncol + (ncol - 1)] );
         }
         else
         {
            vector[n] = stack[stack_top];
         }
      }
      if( !prediction_mode )
      {
         if( isnan( PE[0] ) || isinf( PE[0] ) ) 
            vector[gl_id] = MAXFLOAT;
         else 
            vector[gl_id] = PE[0]/nlin;
      }
   }
}

__kernel void
evaluate_fp( __global const Symbol* phenotype, __global const float* ephemeral, __global const int* size, __global const float* inputs, __global float* vector, __local float* PE, int nlin, int ncol, int prediction_mode, int nInd )
{
   float stack[MAX_PHENOTYPE_SIZE];
   int stack_top;

   int lo_id = get_local_id(0);
   int gr_id = get_group_id(0);
   int gl_id = get_global_id(0);

   int lo_size = get_local_size(0);
   int next_power_of_2 = (pown(2.0f, (int) ceil(log2((float)lo_size))))/2;

   for( int ind = 0; ind < nInd; ++ind )
   {
      barrier(CLK_LOCAL_MEM_FENCE);
   
      if( size[ind] == 0 && !prediction_mode )
      {
         if( lo_id == 0 ) {vector[ind * get_num_groups(0) + gr_id] = MAXFLOAT;}
         continue;
      }

      PE[lo_id] = 0.0f;

      if( gl_id < nlin )
      {
         stack_top = -1;
         for( int i = size[ind] - 1; i >= 0; --i )
         {
            switch( phenotype[ind * MAX_PHENOTYPE_SIZE + i] )
            {
               #include "core"
               case T_ATTRIBUTE:
                  stack[++stack_top] = inputs[(gr_id * lo_size + lo_id) + nlin * (int)ephemeral[ind * MAX_PHENOTYPE_SIZE + i]];
                  break;
               case T_CONST:
                  stack[++stack_top] = ephemeral[ind * MAX_PHENOTYPE_SIZE + i];
                  break;
               default:
                  break;
            }
         }
         if( !prediction_mode )
         {
            PE[lo_id] = fabs( stack[stack_top] - inputs[(gr_id * lo_size + lo_id) + nlin * (ncol - 1)] );
         }
         else
         {
            vector[gl_id] = stack[stack_top];
         }
      }
      if( !prediction_mode )
      {
         for( int s = next_power_of_2; s > 0; s/=2 )
         {
            barrier(CLK_LOCAL_MEM_FENCE);
            if( (lo_id < s) && (lo_id + s < lo_size) ) { PE[lo_id] += PE[lo_id + s]; }
         }
         if( lo_id == 0) { vector[ind * get_num_groups(0) + gr_id] = PE[0]; }
      }
   } 
}

__kernel void
evaluate_ppcu( __global const Symbol* phenotype, __global const float* ephemeral, __global const int* size, __global const float* inputs, __global float* vector, __local float* PE, int nlin, int ncol, int prediction_mode )
{
   float stack[MAX_PHENOTYPE_SIZE];
   int stack_top;

   int lo_id = get_local_id(0);
   int gr_id = get_group_id(0);

   int lo_size = get_local_size(0);
   int next_power_of_2 = (pown(2.0f, (int) ceil(log2((float)lo_size))))/2;
   int n;

   if( size[gr_id] == 0 && !prediction_mode )
   {
      if( lo_id == 0 ) {vector[gr_id] = MAXFLOAT;}
   }
   else
   {
      PE[lo_id] = 0.0f;
      for( int j = 0; j < ceil( nlin/(float) lo_size ); ++j )
      {
         n = j * lo_size + lo_id;
         if( n < nlin )
         {
            stack_top = -1;
            for( int i = size[gr_id] - 1; i >= 0; --i )
            {
               switch( phenotype[gr_id * MAX_PHENOTYPE_SIZE + i] )
               {
                  #include "core"
                  case T_ATTRIBUTE:
                     stack[++stack_top] = inputs[n + nlin * (int)ephemeral[gr_id * MAX_PHENOTYPE_SIZE + i]];
                     break;
                  case T_CONST:
                     stack[++stack_top] = ephemeral[gr_id * MAX_PHENOTYPE_SIZE + i];
                     break;
                  default:
                     break;
               }
            }
            if( !prediction_mode )
            {
               PE[lo_id] += fabs( stack[stack_top] - inputs[n + nlin * (ncol - 1)] );
            }
            else
            {
               vector[n] = stack[stack_top];
            }
         }
      }
      if( !prediction_mode )
      {
         for( int s = next_power_of_2; s > 0; s/=2 )
         {
            barrier(CLK_LOCAL_MEM_FENCE);
            if( (lo_id < s) && (lo_id + s < lo_size) ) { PE[lo_id] += PE[lo_id + s]; }
         }
         if( lo_id == 0)
         { 
            if( isnan( PE[0] ) || isinf( PE[0] ) ) 
               vector[gr_id] = MAXFLOAT;
            else 
               vector[gr_id] = PE[0]/nlin;
         } 
      }
   }
}
