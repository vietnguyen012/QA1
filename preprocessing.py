import csv
import json
import pandas as pd
from pandas import json_normalize    
import tensorflow as tf 

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "inpath", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "outpath", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_bool(
    "istrain", False,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

def json_to_csv(inpath, outpath, istrain=True):
  with open(inpath) as data_file:    
    data = json.load(data_file) 
  if istrain==True:
    json_normalize(data).to_csv(outpath, index=False)
  else:
    json_normalize(data, 'paragraphs', ['__id__','question','title']).to_csv(outpath, index = False)
  return outpath

if __name__ == "__main__":
    json_to_csv(FLAGS.inpath, FLAGS.outpath, FLAGS.istrain)
