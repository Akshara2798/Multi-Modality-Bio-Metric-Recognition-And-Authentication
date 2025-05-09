import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, GlobalMaxPooling1D, Add, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from Optimization import Sim_GHO_Optimization  # Import your custom optimization function

def Sim_OpPTr(input_shape):
    # Pooling Attention Layer with Patch-wise and Embed-wise Attention
    class PoolingAttention(tf.keras.layers.Layer):
        def __init__(self, units):
            super(PoolingAttention, self).__init__()
            self.units = units

        def build(self, input_shape):
            embedding_dim = input_shape[-1]
            
            self.W_patch = self.add_weight(shape=(embedding_dim, self.units),
                                            initializer='random_normal',
                                            trainable=True)
            self.b_patch = self.add_weight(shape=(self.units,),
                                            initializer='zeros',
                                            trainable=True)
            self.u_patch = self.add_weight(shape=(self.units, 1),
                                            initializer='random_normal',
                                            trainable=True)
            
            self.W_embed = self.add_weight(shape=(embedding_dim, self.units),
                                            initializer='random_normal',
                                            trainable=True)
            self.b_embed = self.add_weight(shape=(self.units,),
                                            initializer='zeros',
                                            trainable=True)
            self.u_embed = self.add_weight(shape=(self.units, 1),
                                            initializer='random_normal',
                                            trainable=True)

        def call(self, inputs):
            u_it_patch = tf.tanh(tf.tensordot(inputs, self.W_patch, axes=1) + self.b_patch)
            ait_patch = tf.nn.softmax(tf.tensordot(u_it_patch, self.u_patch, axes=1), axis=1)
            patch_attention_output = tf.reduce_sum(inputs * ait_patch, axis=1)
            
            u_it_embed = tf.tanh(tf.tensordot(inputs, self.W_embed, axes=1) + self.b_embed)
            ait_embed = tf.nn.softmax(tf.tensordot(u_it_embed, self.u_embed, axes=1), axis=2)
            embed_attention_output = tf.reduce_sum(inputs * ait_embed, axis=2)
            
            combined_output = patch_attention_output + embed_attention_output
            
            return combined_output

    # Transformer Encoder Layer using Pooling Attention
    def transformer_encoder_with_pooling_attention(inputs, units, ff_dim, dropout=0.1):
        x = LayerNormalization()(inputs)
        attention_output = PoolingAttention(units=units)(x)
        attention_output = tf.expand_dims(attention_output, axis=1)
        attention_output = Add()([attention_output, inputs])
        x = LayerNormalization()(attention_output)
        x = Dense(ff_dim, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return Add()([x, attention_output])

    
    input_embeddings = Input(shape=input_shape)

    # Transformer Encoder block with Pooling Attention
    x = transformer_encoder_with_pooling_attention(input_embeddings, units=128, ff_dim=368, dropout=0.1)

    # BiLSTM layer
    x = Bidirectional(LSTM(1000, return_sequences=True))(x)

    # Global Max Pooling
    x = GlobalMaxPooling1D()(x)

    model = Model(inputs=input_embeddings, outputs=x)
    
    # Compile the model
    opt = Sim_GHO_Optimization(learning_rate=0.001) 
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model





# %%
class PoolingAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(PoolingAttention, self).__init__()
        self.units = units

    def build(self, input_shape):
        embedding_dim = input_shape[-1]
        
        self.W_patch = self.add_weight(shape=(embedding_dim, self.units),
                                        initializer='random_normal',
                                        trainable=True)
        self.b_patch = self.add_weight(shape=(self.units,),
                                        initializer='zeros',
                                        trainable=True)
        self.u_patch = self.add_weight(shape=(self.units, 1),
                                        initializer='random_normal',
                                        trainable=True)
        
        self.W_embed = self.add_weight(shape=(embedding_dim, self.units),
                                        initializer='random_normal',
                                        trainable=True)
        self.b_embed = self.add_weight(shape=(self.units,),
                                        initializer='zeros',
                                        trainable=True)
        self.u_embed = self.add_weight(shape=(self.units, 1),
                                        initializer='random_normal',
                                        trainable=True)

    def call(self, inputs):
        u_it_patch = tf.tanh(tf.tensordot(inputs, self.W_patch, axes=1) + self.b_patch)
        ait_patch = tf.nn.softmax(tf.tensordot(u_it_patch, self.u_patch, axes=1), axis=1)
        patch_attention_output = tf.reduce_sum(inputs * ait_patch, axis=1)
        
        u_it_embed = tf.tanh(tf.tensordot(inputs, self.W_embed, axes=1) + self.b_embed)
        ait_embed = tf.nn.softmax(tf.tensordot(u_it_embed, self.u_embed, axes=1), axis=2)
        embed_attention_output = tf.reduce_sum(inputs * ait_embed, axis=2)
        
        combined_output = patch_attention_output + embed_attention_output
        
        return combined_output
