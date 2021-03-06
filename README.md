# Parallel Program Induction (PPI) #

Parallel Program Induction (PPI) is a distributed and massively parallel grammatical evolution algorithm for inferring symbolic regression and classification models/programs.

PPI relies on OpenCL to massively parallelize the program evaluations in a portable platform-independent way. The breeding is also parallelized via OpenMP threads. Finally, sockets are used to provide a fault-tolerant communication medium, where different instances of PPI can exchange solutions among them in a semi-isolated fashion.

PPI is a Free Software covered by the [GPLv3+](https://www.gnu.org/licenses/gpl-3.0.en.html) license.

## Building ##

### Requirements ###

 - A C++ compiler (GNU GCC is recommended).
 - [CMake](https://cmake.org) building system.
 - [POCO C++](https://pocoproject.org/) libraries.
 - At least one OpenCL platform should be installed on the machine (an accelerator is recommended but not mandatory). [POCL](http://portablecl.org/) is an open-source OpenCL implementation for CPUs and could be used to provide an OpenCL platform for PPI.


### Grammar assembling ###

~~~~~~~~
   cd build/
   cmake .. -DGRAMMAR=<path to the actual grammar> [-DLABEL=<label>] [-DPoco_DIR=<poco dir>] [-DCMAKE_BUILD_TYPE=<build type>] [-DCOST=<cost matrix>] [-DEF=<error function>] 
~~~~~~~~

For instance:

~~~~~~~~
   cd build/
   cmake .. -DGRAMMAR=../problem/random/grammar.bnf -DLABEL=random -DPoco_DIR=/opt/poco -DCMAKE_BUILD_TYPE=RELEASE
~~~~~~~~

#### Error function and cost matrix ###

The `EF` definition is the formula used to compute the "distance" of a predicted value (`X`) to the corresponding observed value (`Y`). By default it is set as `fabs((X)-(Y))`, that is, simply the absolute error. Other useful formulas are `-DEF='((X)!=(Y))'`, for classification problems, and `-DEF='(((X)-(Y))*((X)-(Y)))'`, for obtaining the squared error.

For classification problems, to define a custom cost matrix one can resort to the `-DCOST` cmake option. Its form is: `-DCOST='[N]={{c00,...,c0N-1},...,{cN-10,...,cN-1N-1}}'`, where `N` is the number of classes (note that class indices should be 0-based) and `cXY` the cost (error) of predicting the class `Y` as `X`. For instance:
  
~~~~~~~~
   cmake .. -DCOST='[4]={{0,1,1,1},{1,0,1,1},{1,1,0,1},{1,1,1,0}}'
~~~~~~~~

In order to use the cost matrix in the error function, one can refer to it by the name `COST` as a 2-dimensional C array. For example:

~~~~~~~~
   cmake .. -DCOST='[4]={{0,1,1,1},{1,0,1,1},{1,1,0,1},{1,1,1,0}}' -DEF='COST[(int)X][(int)Y]'
~~~~~~~~

Note that `X` and `Y` are internally real-valued variables, so it is necessary to cast them to integer (to act as indices). If the cost matrix is defined but the error function is not, then the error function is automatically set as `-DEF='COST[(int)X][(int)Y]'`.


#### Other options (definitions) ###

 - `-DREDUCEMAX=<ON|OFF>`: in the PDP strategy, one can also choose between the sum of the errors (the default) or the maximum error. To enable the latter, just add to cmake the option `-DREDUCEMAX=ON`
 - `-DPROTECTED=<ON|OFF>`: use protected functions (do not generate NaN) instead of unprotected ones
 - `-DNATIVE=<ON|OFF>`: use OpenCL native functions (faster but less precise)

### Compilation ###

~~~~~~~~
   make
~~~~~~~~

## Running (example) ##

### Manually (one instance) ###

~~~~~~~~
   ./main  -v -e -acc -strategy PP -d ../problem/iris/data.csv -port 9080 -cl-d 0 -cl-p 0 -ps 30000 -cp 0.6 -mr 0.01 -ts 5 -nb 2000  -g 1000000 -cl-mls 16 -machine | grep -a '^> [0-9]' | cut -c3- | while read CANDIDATE; do ../script/pareto.py -p pareto.front -dup "$CANDIDATE"; done
~~~~~~~~

The `-dup` option allows duplicate candidates (w.r.t. complexity and error) into the Pareto file, provided that they differ in terms of the symbolic solution (model). Since the actual models differ, strictly speaking one doesn't dominate the other, thus using `-dup` option doesn't violate the principle of non-dominance.

### Through run.py ###

Within `build/`:

~~~~~~~~
   while true; do ../script/run.py -d ../problem/random/data.csv -e ./main -i islands.txt -p 9080 -n 4 -st 5000000 -cl-p 0 -cl-d 0 | grep -a '^> [0-9]' | cut -c3- | while read CANDIDATE; do ../script/pareto.py -p pareto.front -dup "$CANDIDATE"; done; sleep 1; done
~~~~~~~~

This will generate the Pareto front in file 'pareto.front'. Note that you have to run each instance separately (each one having its own port number (`-p <n>`) and ideally on a different accelerator (`-cl-p <p>`, `-cl-d <d>`); for example, you could start each instance on a separate 'screen' tab.

It is possible to pass parameters directly to the PPI's process; in order to do that, use the following convention:

~~~~~~~~
   ... ../script/run.py ... -- <parameters to be passed directly to PPI>
~~~~~~~~

For instance:

~~~~~~~~
   ... ../script/run.py ... -- -min -5.0 -max 5.0
~~~~~~~~

### Using the Pareto front as a repository ###

In this mode, the solutions contained in the Pareto file (actually, their genomes) are reintroduced into new running instances, aiming at the injection of diversity into those instances and, ultimately, improving the Pareto front. The script named `seeder.py` serves this purpose; for example:

~~~~~~~~
   while true; do ../script/seeder.py -i pareto.front -p 9080 -p 9081 -p 9082 -p 9083 -p 9084 -p 9085; sleep 10; done
~~~~~~~~

will periodically (each 10s) send a random genome from the pareto.front file ("Hall of Fame") to each one of the given running instances (`localhost:9080`, ...).


### Watching the Pareto front progress on-the-fly ###

~~~~~~~~
   watch -n 1 "cut -d';' -f2-4 pareto.front|sort -n"
~~~~~~~~

### Putting it all together through 'screen' (AKA doing all above within GNU screen) ###

~~~~~~~~
   # Create the screen session named 'ppi'
   screen -S ppi -A -d -m

   # Create all necessary windows (tabs)
   screen -S ppi -X screen -t island-9080
   screen -S ppi -X screen -t island-9081
   screen -S ppi -X screen -t island-9082
   screen -S ppi -X screen -t island-9083
   screen -S ppi -X screen -t island-9084
   screen -S ppi -X screen -t island-9085

   screen -S ppi -X screen -t seeder
   screen -S ppi -X screen -t pareto

   # Run each island on a different window
   screen -S ppi -p island-9080 -X stuff $'while true; do ../script/run.py -d ../problem/iris/data.csv -e ./iris -i islands.txt -p 9080 -n 1 -st 1000000 -cl-p 0 -cl-d 0 | grep -a \'^> [0-9]\' | cut -c3- | while read CANDIDATE; do ../script/pareto.py -p iris-pareto.front -dup "$CANDIDATE"; done; sleep 1; done\n'
   screen -S ppi -p island-9081 -X stuff $'while true; do ../script/run.py -d ../problem/iris/data.csv -e ./iris -i islands.txt -p 9081 -n 1 -st 2000000 -cl-p 0 -cl-d 0 | grep -a \'^> [0-9]\' | cut -c3- | while read CANDIDATE; do ../script/pareto.py -p iris-pareto.front -dup "$CANDIDATE"; done; sleep 1; done\n'
   screen -S ppi -p island-9082 -X stuff $'while true; do ../script/run.py -d ../problem/iris/data.csv -e ./iris -i islands.txt -p 9082 -n 1 -st 3000000 -cl-p 0 -cl-d 1 | grep -a \'^> [0-9]\' | cut -c3- | while read CANDIDATE; do ../script/pareto.py -p iris-pareto.front -dup "$CANDIDATE"; done; sleep 1; done\n'
   screen -S ppi -p island-9083 -X stuff $'while true; do ../script/run.py -d ../problem/iris/data.csv -e ./iris -i islands.txt -p 9083 -n 1 -st 4000000 -cl-p 0 -cl-d 1 | grep -a \'^> [0-9]\' | cut -c3- | while read CANDIDATE; do ../script/pareto.py -p iris-pareto.front -dup "$CANDIDATE"; done; sleep 1; done\n'
   screen -S ppi -p island-9084 -X stuff $'while true; do ../script/run.py -d ../problem/iris/data.csv -e ./iris -i islands.txt -p 9084 -n 1 -st 5000000 -cl-p 0 -cl-d 2 | grep -a \'^> [0-9]\' | cut -c3- | while read CANDIDATE; do ../script/pareto.py -p iris-pareto.front -dup "$CANDIDATE"; done; sleep 1; done\n'
   screen -S ppi -p island-9085 -X stuff $'while true; do ../script/run.py -d ../problem/iris/data.csv -e ./iris -i islands.txt -p 9085 -n 1 -st 6000000 -cl-p 0 -cl-d 2 | grep -a \'^> [0-9]\' | cut -c3- | while read CANDIDATE; do ../script/pareto.py -p iris-pareto.front -dup "$CANDIDATE"; done; sleep 1; done\n'

   # Runs the seeder (feedback) and visualize the Pareto front
   screen -S ppi -p seeder -X stuff $'while true; do ../script/seeder.py -i iris-pareto.front -p 9080 -p 9081 -p 9082 -p 9083 -p 9084 -p 9085; sleep 5; done\n'
   screen -S ppi -p pareto -X stuff $'watch -n 1 "cut -d\';\' -f2-4 iris-pareto.front|cut -c1-140|sort -n|tail -n 35"\n'
~~~~~~~~

## Running Prediction (example) ##

The prediction mode outputs a prediction for each instance of the dataset (`-d`) with respect to the solution given by its phenotype (`-sol`). This mode does not assume that the dataset has the dependent variable (observation). It can process the instances sequentially; e.g.:

~~~~~~~~
   ./main -d ../problem/vento/speed_A305_1.tes -sol '6 10015 10064 5.000000000000 10035 10015 10064 3.000000000000 10062' -pred
~~~~~~~~

or in parallel if one adds the `-acc` flag and the corresponding strategy (`-strategy`):

~~~~~~~~
   ./main -acc -strategy PP -d ../problem/vento/speed_A305_1.tes -sol '6 10015 10064 5.000000000000 10035 10015 10064 3.000000000000 10062' -pred
~~~~~~~~

## Running Testing Solution (example) ##

In this mode, only the MAE is output. Similarly, it can run sequentially or in parallel:

~~~~~~~~
   ./main -d ../problem/vento/speed_A305_1.tes -sol '6 10015 10064 5.000000000000 10035 10015 10064 3.000000000000 10062'
~~~~~~~~

Since it computes the error, the dependent variable (observation) is required as the last field of the dataset.
