#!/usr/bin/python3

from os.path import join
from absl import flags, app
import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_datasets import parse_function

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing train/val set')
  flags.DEFINE_integer('batch', default = 4, help = 'batch size')

def main(unused_argv):
  trainset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'trainset.tfrecord')).map(parse_function).prefetch(FLAGS.batch).shuffle(FLAGS.batch).batch(FLAGS.batch)
  valset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'testset.tfrecord')).map(parse_function).prefetch(FLAGS.batch).shuffle(FLAGS.batch).batch(FLAGS.batch)
  for graph, label in trainset:
    print(graph.node_sets['atom'].features[tfgnn.HIDDEN_STATE])
    print(graph.edge_sets['bond'].features[tfgnn.HIDDEN_STATE])
    print(label)
    exit()

if __name__ == "__main__":
  add_options()
  app.run(main)
