"""
Test an InceptionV4 stability predictor on the ShapeStacks test split and on
real image data from the FAIR block tower test set.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
import argparse
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import json

sys.path.insert(0, os.environ['SHAPESTACKS_CODE_HOME'])
from tf_models.inception.inception_model import inception_v4_logregr_model_fn, inception_v4_model_fn
from data_provider.shapestacks_provider import shapestacks_input_fn


# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Compute accuracy of an InceptionV4 based stability predictor.')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/tmp/datasets/shapestacks',
    help='The path to the data directory.')
ARGPARSER.add_argument(
    '--split_name', type=str, default='ccs_all',
    help="The name of the split to be used.")
ARGPARSER.add_argument(
    '--model_dir', type=str, default='/tmp/models/stability_predictor',
    help='The directory where the model will be stored.')
ARGPARSER.add_argument(
    '--real_data_dir', type=str, default='',
    help='The path to the real FAIR block tower test set.')
# model parameters
ARGPARSER.add_argument(
    '--display_inputs', type=int, default=0,
    help='The number of input images to display in tensorboard per batch.')
# ARGPARSER.add_argument(
#     '--tfckpt_dir', type=str,
#     help="The directory of the TF model snapshot to use.")
# data augmentation parameters
ARGPARSER.add_argument(
    '--augment', type=str, nargs='+',
    default=['no_augment'],
    help="Apply ImageNet-like training data augmentation.")
# training parameters
ARGPARSER.add_argument(
    '--train_epochs', type=int, default=40,
    help='The number of epochs to train.')
ARGPARSER.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of epochs to run in between evaluations.')
ARGPARSER.add_argument(
    '--batch_size', type=int, default=32,
    help='The number of images per batch.')
ARGPARSER.add_argument(
    '--n_best_eval', type=int, default=5,
    help='Top-N best performing snapshots to keep (according to performance on \
    validation set).')
# memory management parameters
ARGPARSER.add_argument(
    '--memcap', type=float, default=0.8,
    help='Maximum fraction of memory to allocate per GPU.')
ARGPARSER.add_argument(
    '--n_prefetch', type=str, default=32,
    help='How many batches to prefetch into RAM.')


def analyse_checkpoint(dir_snapshot, name_snapshot, unparsed_argv):
  # copy files to FLAGS.model_dir/test_scores
  target_dir = os.path.join(FLAGS.model_dir, 'test_scores_' + name_snapshot)
  print('remove directory ' + target_dir)
  # ignore_errors=1 for dealing with non-existing dir
  shutil.rmtree(target_dir, ignore_errors=1)
  print('copy from directory ' + dir_snapshot)
  print('to directory ' + target_dir)
  shutil.copytree(dir_snapshot, target_dir)

  # set up a RunConfig and the estimator
  gpu_options = tf.GPUOptions(
      allow_growth=True,
      per_process_gpu_memory_fraction=FLAGS.memcap
  )
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  run_config = tf.estimator.RunConfig(
      session_config=sess_config,
  )
  # TODO: set up Estimator with chosen model_fn and run_config

  classifier = tf.estimator.Estimator(
      model_fn=inception_v4_logregr_model_fn,
      model_dir=target_dir,
      config=run_config,
      params={
          'num_display_images': FLAGS.display_inputs
      }
  )

  # TODO: evaluate the model on the corresponding test set
  
  print('\nEvaluating on test set...')
  test_results = classifier.evaluate(
      input_fn=lambda: shapestacks_input_fn(
          mode='eval',
          data_dir=FLAGS.data_dir,
          split_name=FLAGS.split_name,
          batch_size=FLAGS.batch_size,
          num_epochs=FLAGS.epochs_per_eval,
          n_prefetch=int(FLAGS.n_prefetch),
          augment=FLAGS.augment
      ),
      name='test'
  )
  
  # Print test results
  print('\nTest results for snapshot: {}'.format(name_snapshot))
  print('Accuracy: {:.6f}'.format(test_results['accuracy']))
  print('Loss: {:.6f}'.format(test_results['loss']))
  
  # writing accuracies and flags to file
  s_name = FLAGS.split_name
  re_name = 'results_' + s_name
  results = {
    FLAGS.split_name: {
        "sim": float(test_results['accuracy'])
    }
}
  filename = "results_" + s_name + ".py"
  filename = os.path.join(target_dir, filename)
  with open(filename, "w") as f:
    f.write(f"{re_name} = " + repr(results) + "\n")

def main(unparsed_argv):
  """
  Pseudo-main executed via tf.app.run().
  """
  # using the Winograd non-fused algorithms provides a small performance boost
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  snapshot_types = ['real', 'eval']
  for snapshot_type in snapshot_types:
    search_dir = os.path.join(FLAGS.model_dir, 'snapshots')

    # Check if snapshots directory exists
    if not os.path.exists(search_dir):
      print(f'WARNING: Snapshots directory does not exist: {search_dir}')
      print(f'Skipping snapshot_type: {snapshot_type}')
      continue
    # List all subdirectories
    all_subdirs = [name for name in os.listdir(search_dir) \
        if os.path.isdir(os.path.join(search_dir, name))]
    print(f'\nAll subdirectories in {search_dir}:')
    for subdir in all_subdirs:
      print(f'  - {subdir}')
    
    # Filter by snapshot_type
    list_subdir = [name for name in all_subdirs if snapshot_type in name]
    
    if not list_subdir:
      print(f'WARNING: No snapshots found for type "{snapshot_type}"')
      print(f'Skipping snapshot_type: {snapshot_type}')
      continue

    # list_subdir = [name for name in os.listdir(search_dir) \
    #     if os.path.isdir(os.path.join(search_dir, name)) \
    #     and snapshot_type in name]
    # list_subdir = [x for _, x in sorted(zip([float(name[5:]) \
    #     for name in list_subdir], list_subdir), reverse=True)]

    def extract_score(name):
      if '=' in name:
        return float(name.split('=')[1])
      else:
        # Original format: assuming name starts with 'eval_' or 'real_'
        return float(name[5:])
    list_subdir = [x for _, x in sorted(
        zip([extract_score(name) for name in list_subdir], list_subdir), 
        reverse=True)]
    print(f'\nSnapshots for type "{snapshot_type}" (sorted):')
    
    for i in list_subdir:
      print(i + '\n')
    name_snapshot = list_subdir[0]
    dir_snapshot = os.path.join(FLAGS.model_dir, 'snapshots', name_snapshot)
    analyse_checkpoint(dir_snapshot, name_snapshot, unparsed_argv)


if __name__ == '__main__':
  FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
  print("FLAGS:", FLAGS)
  print("UNPARSED_ARGV:", UNPARSED_ARGV)
  tf.app.run(argv=[sys.argv[0]] + UNPARSED_ARGV)
