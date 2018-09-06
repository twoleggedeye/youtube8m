# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class EnsembleModelA(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   **unused_params):
    x = slim.fully_connected(
        model_input,
        1024,
        activation_fn=None,
        biases_initializer=None,
        scope="model_a_fc1")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_a_bn1")

    x = tf.nn.sigmoid(x) * x

    x = slim.fully_connected(
        x,
        1024,
        activation_fn=None,
        biases_initializer=None,
        scope="model_a_fc2")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_a_bn2")

    x = tf.nn.sigmoid(x) * x

    output = slim.fully_connected(
        x,
        1024,
        activation_fn=None,
        biases_initializer=None,
        scope="model_a_fc3")

    return {"predictions": tf.nn.relu(output)}


class EnsembleModelB(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   **unused_params):
    x = slim.fully_connected(
        model_input,
        1024,
        activation_fn=None,
        biases_initializer=None,
        scope="model_b_fc1")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_b_bn1")

    x = tf.nn.relu(x)

    output = slim.fully_connected(
        x,
        1024,
        activation_fn=None,
        biases_initializer=None,
        scope="model_b_fc2")

    return {"predictions": tf.nn.relu(output)}


class EnsembleModelC(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   **unused_params):
    input_video = model_input[:, :1024]
    input_audio = model_input[:, 1024:]

    input_video = slim.fully_connected(
        input_video,
        1024,
        activation_fn=None,
        biases_initializer=None,
        scope="model_c_fc1")

    input_video = slim.batch_norm(
        input_video,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_c_bn1")

    input_video = tf.nn.leaky_relu(input_video)

    input_audio = slim.fully_connected(
        input_audio,
        256,
        activation_fn=None,
        biases_initializer=None,
        scope="model_c_fc2")

    input_audio = slim.batch_norm(
        input_audio,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_c_bn2")

    input_audio = tf.nn.leaky_relu(input_audio)

    x = tf.concat([input_video, input_audio], 1)

    output = slim.fully_connected(
        x,
        1024,
        activation_fn=None,
        biases_initializer=None,
        scope="model_c_fc3")

    return {"predictions": tf.nn.relu(output)}


class EnsembleModelD(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     **unused_params):
        input_video = model_input[:, :1024]
        input_audio = model_input[:, 1024:]

        input_video = slim.fully_connected(
            input_video,
            1024,
            activation_fn=None,
            biases_initializer=None,
            scope="model_d_fc1")

        input_video = slim.batch_norm(
            input_video,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="model_d_bn1")

        input_video = tf.nn.sigmoid(input_video) * input_video

        input_video = slim.fully_connected(
            input_video,
            1024,
            activation_fn=None,
            biases_initializer=None,
            scope="model_d_fc2")

        input_video = slim.batch_norm(
            input_video,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="model_d_bn2")

        input_video = tf.nn.sigmoid(input_video) * input_video

        input_audio = slim.fully_connected(
            input_audio,
            256,
            activation_fn=None,
            biases_initializer=None,
            scope="model_d_fc3")

        input_audio = slim.batch_norm(
            input_audio,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="model_d_bn3")

        input_audio = tf.nn.sigmoid(input_audio) * input_audio

        input_audio = slim.fully_connected(
            input_audio,
            256,
            activation_fn=None,
            biases_initializer=None,
            scope="model_d_fc4")

        input_audio = slim.batch_norm(
            input_audio,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="model_d_bn4")

        input_audio = tf.nn.sigmoid(input_audio) * input_audio

        x = tf.concat([input_video, input_audio], 1)

        output = slim.fully_connected(
            x,
            1024,
            activation_fn=None,
            biases_initializer=None,
            scope="model_d_fc5")

        return {"predictions": tf.nn.relu(output)}


class EnsembleModelE(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   **unused_params):
    x = slim.fully_connected(
        model_input,
        4096,
        activation_fn=None,
        biases_initializer=None,
        scope="model_e_fc1")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_e_bn1")

    x = tf.nn.relu(x)

    x = slim.fully_connected(
        x,
        4096,
        activation_fn=None,
        biases_initializer=None,
        scope="model_e_fc2")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_e_bn2")

    x = tf.nn.relu(x)

    x = slim.fully_connected(
        x,
        4096,
        activation_fn=None,
        biases_initializer=None,
        scope="model_e_fc3")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_e_bn3")

    x = tf.nn.relu(x)

    x = slim.fully_connected(
        x,
        4096,
        activation_fn=None,
        biases_initializer=None,
        scope="model_e_fc4")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_e_bn4")

    x = tf.nn.relu(x)

    output = slim.fully_connected(
        x,
        2048,
        activation_fn=None,
        biases_initializer=None,
        scope="model_e_fc5")

    return {"predictions": tf.nn.relu(output)}



class EnsembleModelF(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   **unused_params):
    x = slim.fully_connected(
        model_input,
        2048,
        activation_fn=None,
        biases_initializer=None,
        scope="model_f_fc1")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_f_bn1")

    x = tf.nn.relu(x)

    x = slim.fully_connected(
        x,
        2048,
        activation_fn=None,
        biases_initializer=None,
        scope="model_f_fc2")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_f_bn2")

    x = tf.nn.relu(x)

    x = slim.fully_connected(
        x,
        2048,
        activation_fn=None,
        biases_initializer=None,
        scope="model_f_fc3")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_f_bn3")

    x = tf.nn.relu(x)

    x = slim.fully_connected(
        x,
        2048,
        activation_fn=None,
        biases_initializer=None,
        scope="model_f_fc4")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_f_bn4")

    x = tf.nn.relu(x)

    x = slim.fully_connected(
        x,
        2048,
        activation_fn=None,
        biases_initializer=None,
        scope="model_f_fc5")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="model_f_bn5")

    x = tf.nn.relu(x)

    output = slim.fully_connected(
        x,
        2048,
        activation_fn=None,
        biases_initializer=None,
        scope="model_f_fc6")

    return {"predictions": tf.nn.relu(output)}


class EnsembleModel(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   **unused_params):
    predictions = []
    predictions.append(EnsembleModelA().create_model(model_input, vocab_size)["predictions"])
    predictions.append(EnsembleModelB().create_model(model_input, vocab_size)["predictions"])
    predictions.append(EnsembleModelC().create_model(model_input, vocab_size)["predictions"])
    predictions.append(EnsembleModelD().create_model(model_input, vocab_size)["predictions"])
    predictions.append(EnsembleModelE().create_model(model_input, vocab_size)["predictions"])
    predictions.append(EnsembleModelF().create_model(model_input, vocab_size)["predictions"])
    predictions.append(model_input)

    x = tf.concat(predictions, 1)
    x = slim.fully_connected(
        x,
        8192,
        activation_fn=None,
        biases_initializer=None,
        scope="ensemble_fc1")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="ensemble_bn1")

    x = tf.nn.relu(x)

    x = slim.fully_connected(
        x,
        6000,
        activation_fn=None,
        biases_initializer=None,
        scope="ensemble_fc2")

    x = slim.batch_norm(
        x,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="ensemble_bn2")

    x = tf.nn.relu(x)

    output = slim.fully_connected(
        x,
        vocab_size,
        activation_fn=None,
        biases_initializer=None,
        scope="ensemble_output")

    output = tf.nn.sigmoid(output)

    return {"predictions": output}
