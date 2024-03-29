#!/usr/bin/env python3

import os, sys
import argparse
import time
import subprocess
try:
   import filelock
   lock = filelock.FileLock
except: # Fallback to simple lock (not as good, but...)
   print("> Warning: fallbacking to module simpleflock, but module filelock would be better.", file=sys.stderr)
   import simpleflock
   lock = simpleflock.SimpleFlock

# Format: 'generation;size;error;solution;phenotype;genome;command-line;timings;...'
#                    |_________|
FIELD_START=1
FIELD_END=3

def IsValid(p):
   if len(p.split(';')) < 2: # Must has at least two delimiters (;)
      return False
   try: # FIELD_START up to FIELD_END must be convertible to float
      for value in p.split(';')[FIELD_START:FIELD_END]:
         float(value)
   except:
      return False
   return True

def UpdatePareto(pareto, candidate, FIELD_END_DUPLICATE):
   if len(pareto) == 0: # Empty pareto, obviously 'candidate' dominates the empty
      return True, [candidate]

   # Check if the candidate is dominated by any of the current members; if so, then bye bye...
   cmd = [os.path.dirname(os.path.realpath(__file__))+"/"+"domination-many.py", "-t", "less", "-a", "dominated", ','.join(candidate.split(';')[FIELD_START:FIELD_END])]
   for p in pareto:
      if p.split(';')[FIELD_START:FIELD_END_DUPLICATE] == candidate.split(';')[FIELD_START:FIELD_END_DUPLICATE]:
         return False, pareto # candidate is already a pareto front member

      complexity_error = "%s\n"%(','.join(p.split(';')[FIELD_START:FIELD_END]))
      domination = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
      output, error = domination.communicate( complexity_error.encode() )
      if error is not None:
         print(error, file=stderr)
      if domination.returncode == 0: # candidate is dominated by someone
         return False, pareto

   # Hmmm, dominated by none of them, it deserves inclusion as a new member of the pareto front
   print("\n> New pareto front member: [%s]" % (';'.join(candidate.split(';')[FIELD_START:FIELD_END])))

   new_pareto = []
   # Now, check if one or more members will leave the pareto front because of the introduction of candidate
   cmd = [os.path.dirname(os.path.realpath(__file__))+"/"+"domination-many.py", "-t", "less", "-a", "dominates", ','.join(candidate.split(';')[FIELD_START:FIELD_END])]
   for p in pareto:
      complexity_error = "%s\n"%(','.join(p.split(';')[FIELD_START:FIELD_END]))
      domination = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
      output, error = domination.communicate( complexity_error.encode() )
      if error is not None:
         print(error, file=stderr)
      if domination.returncode == 1: # candidate does not dominate p
         new_pareto.append(p)
      else:
         print("\n> Member [%s] has left the pareto front due to the domination of the new member [%s]" % (';'.join(p.split(';')[FIELD_START:FIELD_END]),';'.join(candidate.split(';')[FIELD_START:FIELD_END])))

   # Add the new non-dominated member at the end of the list
   new_pareto.append(candidate)

   return True, new_pareto

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("-p", "--pareto-file", required=True, help="Pareto file path")
   parser.add_argument("-l", "--lock-file", required=False, default="pareto.lock", help="Lock file path")
   parser.add_argument("-dup", "--allow-duplicate", action='store_true', default=False, help="Allow duplicate entries provided they have different symbolic solutions")
   parser.add_argument("candidate", help="Candidate")
   args = parser.parse_args()

   # Some validation regarding the given candidate
   if not IsValid(args.candidate):
      print("[INVALID Candidate: %s]" % (args.candidate), file=sys.stderr)
      sys.exit(0)

   print("[Candidate: %s]" % (args.candidate), file=sys.stderr)

   if args.allow_duplicate: # Includes the "solution" itself in order to whether consider duplicate or not
      FIELD_END_DUPLICATE=FIELD_END+1
   else:
      FIELD_END_DUPLICATE=FIELD_END

   #sys.stdout.write("(Processing: [%s]...)" % (';'.join(args.candidate.split(';')[FIELD_START:FIELD_END])))

   old_pareto = []; new_pareto = []; changed = None

   # A lock is required here since other pareto.py process might be writing to the pareto_file right now
   with lock(args.lock_file):
      # Create the pareto file if it doesn't exist:
      with open(args.pareto_file, 'a+') as pareto:
         pass

      with open(args.pareto_file) as pareto:
         old_pareto = pareto.read().splitlines()
         # Strangely enough, sometimes malformed entries end up into the Pareto
         # file. This procedure ignores such dirty entries.
         for p in old_pareto:
            if not IsValid(p): old_pareto.remove(p)

   changed, new_pareto = UpdatePareto(old_pareto, args.candidate, FIELD_END_DUPLICATE)

   #print old_pareto, new_pareto

#time.sleep(10)
   if changed:
      #print "New pareto front member: %s" % (args.candidate)
      cur_pareto = []
      with lock(args.lock_file):
         #print "lock acquired!"
         with open(args.pareto_file, 'r+') as pareto:
            cur_pareto = pareto.read().splitlines()
            # Strangely enough, sometimes malformed entries end up into the Pareto
            # file. This procedure ignores such dirty entries.
            for p in cur_pareto:
               if not IsValid(p): cur_pareto.remove(p)
            pareto.seek(0)
            if cur_pareto == old_pareto:
               #print "Equal"
               #print '\n'.join(new_pareto)
               pareto.truncate()
               pareto.write('\n'.join(new_pareto)+"\n")
            else: # Hmmm, someone else wrote to the pareto_file in the meantime, recalculating the pareto...
               #print "Different"
               changed, new_pareto = UpdatePareto(cur_pareto, args.candidate, FIELD_END_DUPLICATE)
               if changed:
                  #print "New: ", new_pareto
                  pareto.truncate()
                  pareto.write('\n'.join(new_pareto)+"\n")
#
## Raises an IOError in 3 seconds if unable to acquire the lock.
#with simpleflock.SimpleFlock("/tmp/foolock", timeout = 3):
#   # Do something.
#   pass
main()
