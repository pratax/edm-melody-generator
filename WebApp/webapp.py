# Copyright 2022 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate melodies from a trained checkpoint of a melody RNN model."""
import ast
import os
import time
import sys

sys.path.insert(0, "../magenta")

from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.melody_rnn import melody_rnn_model
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator
from magenta.models.shared import sequence_generator_bundle
import note_seq
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
tf.app.flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
tf.app.flags.DEFINE_string(
    'bundle_file', None,
    'Path to the bundle file. If specified, this will take priority over '
    'run_dir and checkpoint_file, unless save_generator_bundle is True, in '
    'which case both this flag and either run_dir or checkpoint_file are '
    'required')
tf.app.flags.DEFINE_boolean(
    'save_generator_bundle', False,
    'If true, instead of generating a sequence, will save this generator as a '
    'bundle file in the location specified by the bundle_file flag')
tf.app.flags.DEFINE_string(
    'bundle_description', None,
    'A short, human-readable text description of the bundle (e.g., training '
    'data, hyper parameters, etc.).')
tf.app.flags.DEFINE_string(
    'output_dir', '../tmp/melody_rnn/generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', 1,
    'The number of melodies to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_integer(
    'num_steps', 128,
    'The total number of steps the generated melodies should be, priming '
    'melody length + generated steps. Each step is a 16th of a bar.')
tf.app.flags.DEFINE_string(
    'primer_melody', '', 'A string representation of a Python list of '
    'note_seq.Melody event values. For example: '
    '"[60, -2, 60, -2, 67, -2, 67, -2]". If specified, this melody will be '
    'used as the priming melody. If a priming melody is not specified, '
    'melodies will be generated from scratch.')
tf.app.flags.DEFINE_string(
    'primer_midi', '',
    'The path to a MIDI file containing a melody that will be used as a '
    'priming melody. If a primer melody is not specified, melodies will be '
    'generated from scratch.')
tf.app.flags.DEFINE_float(
    'qpm', None,
    'The quarters per minute to play generated output at. If a primer MIDI is '
    'given, the qpm from that will override this flag. If qpm is None, qpm '
    'will default to 120.')
tf.app.flags.DEFINE_float(
    'temperature', 1.0,
    'The randomness of the generated melodies. 1.0 uses the unaltered softmax '
    'probabilities, greater than 1.0 makes melodies more random, less than 1.0 '
    'makes melodies less random.')
tf.app.flags.DEFINE_integer(
    'beam_size', 1,
    'The beam size to use for beam search when generating melodies.')
tf.app.flags.DEFINE_integer(
    'branch_factor', 1,
    'The branch factor to use for beam search when generating melodies.')
tf.app.flags.DEFINE_integer(
    'steps_per_iteration', 1,
    'The number of melody steps to take per beam search iteration.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')


def get_checkpoint():
  """Get the training dir or checkpoint path to be used by the model."""
  if ((FLAGS.run_dir or FLAGS.checkpoint_file) and
      FLAGS.bundle_file and not FLAGS.save_generator_bundle):
    raise sequence_generator.SequenceGeneratorError(
        'Cannot specify both bundle_file and run_dir or checkpoint_file')
  if FLAGS.run_dir:
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')
    return train_dir
  elif FLAGS.checkpoint_file:
    return os.path.expanduser(FLAGS.checkpoint_file)
  else:
    return None


def get_bundle():
  """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.
  Returns:
    Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
    not set or the save_generator_bundle flag is set.
  """
  bundle_file = os.path.expanduser("WebApp/run5.mag")
  return sequence_generator_bundle.read_bundle_file(bundle_file)


def run_with_flags(generator, primer_melody, temperature, length, qpm, primer_midi):
  """Generates melodies and saves them as MIDI files.
  Uses the options specified by the flags defined in this module.
  Args:
    generator: The MelodyRnnSequenceGenerator to use for generation.
  """
  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return
  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)

  #primer_midi = None
  if primer_midi:
    primer_midi = os.path.expanduser(primer_midi)

  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  primer_sequence = None
  #qpm = FLAGS.qpm if FLAGS.qpm else note_seq.DEFAULT_QUARTERS_PER_MINUTE
  qpm = qpm

  if primer_midi:
    primer_sequence = note_seq.midi_file_to_sequence_proto(primer_midi)
    if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
      qpm = primer_sequence.tempos[0].qpm
  elif primer_melody:
    primer_melody = note_seq.Melody(ast.literal_eval(primer_melody))
    primer_sequence = primer_melody.to_sequence(qpm=qpm)
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to a single middle C.')
    primer_melody = note_seq.Melody([60])
    primer_sequence = primer_melody.to_sequence(qpm=qpm)

  # Derive the total number of seconds to generate based on the QPM of the
  # priming sequence and the num_steps flag.
  seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
  #total_seconds = FLAGS.num_steps * seconds_per_step
  total_seconds = length * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  if primer_sequence:
    input_sequence = primer_sequence
    # Set the start time to begin on the next step after the last note ends.
    if primer_sequence.notes:
      last_end_time = max(n.end_time for n in primer_sequence.notes)
    else:
      last_end_time = 0
    generate_section = generator_options.generate_sections.add(
        start_time=last_end_time + seconds_per_step,
        end_time=total_seconds)

    if generate_section.start_time >= generate_section.end_time:
      tf.logging.fatal(
          'Priming sequence is longer than the total number of steps '
          'requested: Priming sequence length: %s, Generation length '
          'requested: %s',
          generate_section.start_time, total_seconds)
      return
  else:
    input_sequence = music_pb2.NoteSequence()
    input_sequence.tempos.add().qpm = qpm
    generate_section = generator_options.generate_sections.add(
        start_time=0,
        end_time=total_seconds)
  generator_options.args['temperature'].float_value = temperature
  generator_options.args['beam_size'].int_value = FLAGS.beam_size
  generator_options.args['branch_factor'].int_value = FLAGS.branch_factor
  generator_options.args[
      'steps_per_iteration'].int_value = FLAGS.steps_per_iteration
  tf.logging.debug('input_sequence: %s', input_sequence)
  tf.logging.debug('generator_options: %s', generator_options)

  # Make the generate request num_outputs times and save the output as midi
  # files.
  date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
  digits = len(str(FLAGS.num_outputs))
  generated_sequence = generator.generate(input_sequence, generator_options)

  midi_filename = '%s_%s.mid' % (date_and_time, str(0 + 1).zfill(digits))
  midi_path = os.path.join(FLAGS.output_dir, midi_filename)
  note_seq.sequence_proto_to_midi_file(generated_sequence, midi_path)

  tf.logging.info('Wrote %d MIDI files to %s',
                  FLAGS.num_outputs, FLAGS.output_dir)

  return midi_path


def main(primer_melody, temperature, length, qpm, primer_midi):
  """Saves bundle or runs generator based on flags."""
  tf.logging.set_verbosity(FLAGS.log)

  bundle = get_bundle()

  config_id = bundle.generator_details.id
  config = melody_rnn_model.default_configs[config_id]
  config.hparams.parse(FLAGS.hparams)

  generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
      model=melody_rnn_model.MelodyRnnModel(config),
      details=config.details,
      steps_per_quarter=config.steps_per_quarter,
      checkpoint=get_checkpoint(),
      bundle=bundle)
  midi_path = run_with_flags(generator, primer_melody, temperature, length, qpm, primer_midi)
  return midi_path


def console_entry_point(primer_melody, temperature, length, qpm, primer_midi=None):
  tf.disable_v2_behavior()
  midi_path = main(primer_melody, temperature, length, qpm, primer_midi)
  return midi_path


if __name__ == '__main__':
  import streamlit as st
  import pandas as pd
  import pretty_midi
  from scipy.io import wavfile
  import pygame


  def play_music(midi_filename):
      '''Stream music_file in a blocking manner'''
      clock = pygame.time.Clock()
      pygame.mixer.music.load(midi_filename)
      pygame.mixer.music.play()
      while pygame.mixer.music.get_busy():
          clock.tick(30)  # check if playback has finished

  st.title('edm-melody-generator :notes: :dancer:')
  st.sidebar.title("Settings")

  st.write("Here's our first attempt at using data to create a table:")
  temperature = st.sidebar.slider('Randomness', 0.1, 10.0, value=1.0)  # 👈 this is a widget
  qpm = st.sidebar.number_input("BPM", min_value=40, max_value=200, value=120, step=1, format="%i")
  bars = st.sidebar.select_slider("Bars", options=[1, 4, 8, 16], value=8)
  length = bars*16

  df = pd.DataFrame({
      'first column': ['C','D','E','F','G','A','B'],
  })

  key_to_primer = {
      'C': "[60]",
      'D': "[62]",
      'E': "[64]",
      'F': "[65]",
      'G': "[67]",
      'A': "[69]",
      'B': "[71]",
  }

  option = st.sidebar.selectbox('Key: ', df['first column'])

  #midi_data = pretty_midi.PrettyMIDI(st.sidebar.file_uploader("Choose a MIDI file to start the melody with", type=['mid']))
  #midi_path = os.path.join(os.getcwd(), 'upload.mid')
  #midi_data.write(midi_path)

  if st.button('Generate Melody'):

      #st.button("generate", key=None, help=None, on_click=console_entry_point(), args=None, kwargs=None, type="secondary", disabled=False)
      midi_file = console_entry_point(primer_melody=key_to_primer[option], temperature=temperature, length=length, qpm=qpm, primer_midi=None)
      # mixer config
      freq = 44100  # audio CD quality
      bitsize = -16  # unsigned 16 bit
      channels = 2  # 1 is mono, 2 is stereo
      buffer = 1024  # number of samples
      pygame.mixer.init(freq, bitsize, channels, buffer)

      # optional volume 0 to 1.0
      pygame.mixer.music.set_volume(0.8)


      def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
          # Use librosa's specshow function for displaying the piano roll
          librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                                   hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                                   fmin=pretty_midi.note_number_to_hz(start_pitch))


      import matplotlib.pyplot as plt
      import librosa.display
      pianoroll = plt.figure(figsize=(8, 4))
      plot_piano_roll(pretty_midi.PrettyMIDI(midi_file), 55, 80)
      st.pyplot(pianoroll)

      with st.spinner(f"Playing the generated melody..."):
          play_music(midi_file)

