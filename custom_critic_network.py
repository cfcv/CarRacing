import tensorflow as tf

from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils


# def validate_specs(action_spec, observation_spec):
#     """Validates the spec contains a single action."""
#     del observation_spec  # not currently validated

#     flat_action_spec = tf.nest.flatten(action_spec)
#     if len(flat_action_spec) > 1:
#         raise ValueError("Network only supports action_specs with a single action.")

#     if flat_action_spec[0].shape not in [(), (1,)]:
#         raise ValueError("Network only supports action_specs with shape in [(), (1,)])")


class CriticNetwork(network.Network):
    """Creates a critic network."""

    def __init__(
        self,
        input_tensor_spec,
        observation_preprocessing_layers=None,
        observation_preprocessing_combiner=None,
        observation_conv_layer_params=None,
        observation_fc_layer_params=(200, 200),
        observation_dropout_layer_params=None,
        action_fc_layer_params=None,
        action_dropout_layer_params=None,
        joint_fc_layer_params=None,
        joint_dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu,
        kernel_initializer=None,
        batch_squash=True,
        dtype=tf.float32,
        name="CriticNetwork",
    ):

        super(CriticNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )

        encoder_input_tensor_spec, _ = input_tensor_spec


        self._encoder = encoding_network.EncodingNetwork(
            encoder_input_tensor_spec,
            preprocessing_layers=observation_preprocessing_layers,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            dropout_layer_params=observation_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype,
            name="observation_encoding"
        )

        self._action_layers = utils.mlp_layers(
            conv_layer_params=None,
            fc_layer_params=action_fc_layer_params,
            dropout_layer_params=action_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
            ),
            name="action_encoding",
        )

        self._joint_layers = utils.mlp_layers(
            conv_layer_params=None,
            fc_layer_params=joint_fc_layer_params,
            dropout_layer_params=joint_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
            ),
            name="joint_mlp",
        )

        self._joint_layers.append(
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.003, maxval=0.003
                ),
                name="value",
            )
        )

    def call(self, inputs, step_type=(), network_state=(), training=False):
        observations, actions = inputs
        
        observations = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
        observations, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state
        )

        actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)
        for layer in self._action_layers:
            actions = layer(actions, training=training)

        joint = tf.concat([observations, actions], 1)
        for layer in self._joint_layers:
            joint = layer(joint, training=training)

        return tf.reshape(joint, [-1]), network_state
