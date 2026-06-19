"""
#########################################################################################################
            Converting CSV to libsvm Data Format
#########################################################################################################
"""
import argparse
import sys
import csv
import operator
from collections import defaultdict
label_index = 0

def construct_line( label, line ): 
  new_line = [] 
  if float( label ) == 0.0: 
    label = "0" 
  new_line.append( label )

  for i, item in enumerate( line ): 
    if len(item.strip())==0 or float( item ) == 0.0: 
      continue # sparse!!! 
    new_item = "%s:%s" % ( i + 1, item ) 
    new_line.append( new_item ) 
  new_line = " ".join( new_line ) 
  new_line += "\n" 
  return new_line

def csv2libsvm(input_file,output_file):
  i = open( input_file, "r")
  o = open( output_file, 'w' )
  reader = csv.reader( i )  
  for line in reader:
    if label_index == -1:
      label = '1'
    else:
      label = line.pop( label_index )
    new_line = construct_line( label, line )
    o.write( new_line )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='csv_dir/', help='csvs to convert')
    parser.add_argument('--output_dir', type=str, default='libsvm_dir/', help='output file will have the same name of the csv without the extension')
    opt = parser.parse_args()
    return opt

    
opt = parse_opt()

output_dir = opt.output_dir
csv_file = opt.csv_file
libsvm_file = csv_file.split('/')[-1].replace('.csv', '')

csv2libsvm(csv_file, output_dir + '/' + libsvm_file)
