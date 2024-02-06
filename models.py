#!/usr/bin/python3

import tensorflow as tf
import tensorflow_gnn as tfgnn
from create_datasets import graph_tensor_spec

def FeatureExtract(channels = 256, layer_num = 4, drop_rate = 0.5):
  inputs = tf.keras.Input(type_spec = graph_tensor_spec())
  results = inputs.merge_batch_to_components() # merge graphs of a batch to one graph as different components
  results = tfgnn.keras.layers.MapFeatures(
    node_sets_fn = lambda node_set, *, node_set_name: tf.keras.layers.Dense(channels)(node_set[tfgnn.HIDDEN_STATE]),
    edge_sets_fn = lambda edge_set, *, edge_set_name: tf.keras.layers.Dense(channels)(edge_set[tfgnn.HIDDEN_STATE]))(results)
  # only update node vectors
  for i in range(layer_num):
    results = tfgnn.keras.layers.GraphUpdate(
      node_sets = {
        "atom": tfgnn.keras.layers.NodeSetUpdate(
          edge_set_inputs = {
            "bond": tfgnn.keras.layers.SimpleConv(
              message_fn = tf.keras.Sequential([
                tf.keras.layers.Dense(channels, activation = tf.keras.activations.gelu, kernel_regularizer = tf.keras.regularizers.l2(5e-4), bias_regularizer = tf.keras.regularizers.l2(5e-4)),
                tf.keras.layers.Dropout(drop_rate)
              ]),
              reduce_type = "sum",
              receiver_tag = tfgnn.TARGET
            )
          },
          next_state = tfgnn.keras.layers.NextStateFromConcat(
            transformation = tf.keras.Sequential([
              tf.keras.layers.Dense(channels, activation = tf.keras.activations.gelu, kernel_regularizer = tf.keras.regularizers.l2(5e-4), bias_regularizer = tf.keras.regularizers.l2(5e-4)),
              tf.keras.layers.Dropout(drop_rate)
            ])
          )
        )
      }
    )(results)
  # graph pooling
  results = tfgnn.keras.layers.Pool(tag = tfgnn.CONTEXT, reduce_type = "mean", node_set_name = "atom")(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

def Predictor(channels = 256, layer_num = 4, drop_rate = 0.5):
  inputs = tf.keras.Input(type_spec = graph_tensor_spec())
  results = FeatureExtract(channels, layer_num, drop_rate)(inputs)
  results = tf.keras.layers.Dense(2, activation = tf.keras.activations.softmax)(results)
  return tf.keras.Model(inputs = inputs, outputs = results)
