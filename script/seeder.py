#!/usr/bin/env python3

import socket, argparse, sys, csv, random

"""
   This script when invoked will send a random genome (given a bank of genomes)
   to all specified peers in '-p'

   Usage (ex):
      ./seeder.py -i pareto.front -p 9080 -p remote:22257
"""

parser = argparse.ArgumentParser()

parser.add_argument('-i', dest='input', help='Input file (seeds)')
parser.add_argument('-f', dest='field', type=int, default=6, help='Field id')
parser.add_argument('-d', dest='delimiter', default=';', help='Delimiter')
parser.add_argument('-p', action='append', dest='peers', default=[], help='Peer (server) which will receive the seeds')

args = parser.parse_args()

genomes = []
with open(args.input) as i:
   pareto = csv.reader(i, delimiter=args.delimiter)
   for member in pareto:
      genomes.append(member[args.field-1])

for peer in args.peers:
   address = None
   port = None

   if ':' in peer:
      address, port = peer.split(':')
   else:
      port = peer

   try:
      port = int(port)
   except:
      print("Invalid port: %s" % (peer), file=sys.stderr)

   if address is None or len(address) == 0:
      address = 'localhost'

   #print address, port

   try:
      genome = random.choice(genomes)
      message = "0.0 " + genome # = "fitness" + genome
      sock = socket.socket( socket.AF_INET, socket.SOCK_STREAM )
      sock.connect( (address,port) )
      # Send the header (Ex: "I000000023")
      header = 'I%s' % (str(len(message)).rjust(9, '0'))
      print("[%s:%d] Trying to send header: %s" % (address,port,header), file=sys.stderr)
      if sock.send(header.encode()) != 10:
         raise Exception()
      # Send the message
      print("[%s:%d] Trying to send message of size %dB: [%s]" % (address,port,len(message),message), file=sys.stderr)
      if sock.send(message.encode()) != len(message):
         raise Exception()
   except Exception as e:
      print("[%s:%d] Couldn't communicate: %s" % (address,port,e), file=sys.stderr)
   else:
      print("[%s:%d] Sent genome" % (address,port))
