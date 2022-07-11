from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Input, MaxPool2D, Concatenate, Dropout, ZeroPadding2D, Cropping2D
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose, Deconv2D
import tensorflow as tf

class Model_Architecture_Builder:
	'''
	This utility class builds the default unet architecture for all models to avoid repeated code.
	'''

	@staticmethod
	def create_default_model() -> tf.keras.Model:
		'''
		This static method is responsible for creating the default unet architecture and returning it.

		source: https://medium.com/analytics-vidhya/unet-implementation-in-tensorflow-using-keras-api-idiot-developer-bc3504e9ca69
		
		Parameters
		----------
		None

		Returns
		-------
		tf.keras.Model

		Raises
		------
		None
		'''

		INPUT_SHAPE = (1025, 173, 1) #default input spectrogram shape (for 2 second segments). First dimension is the height, second is the width, third is the number of channels

		print("Building the default Unet architecture")




		def build_Unet() -> tf.keras.Model:
			'''
			The following creates a default unet model
			'''

			print(f"Building the default Unet architecture with input_shape: {INPUT_SHAPE}")

			class Conv_Block(tf.keras.Model):
				'''
				This class is responsible for creating a convolutional block of the downsampling path.
				
				It has the following layers:
				
				1. conv2d
				2. activation
				3. batch normalization
				5. activation				
				6. batch normalization

				'''
				def __init__(self, num_filters:int, block_name:str):

					super(Conv_Block, self).__init__()
					self.num_filters = num_filters
					self.block_name = block_name
					self.conv2d_1 = Conv2D(filters=self.num_filters,
											kernel_size=3,
											strides=(1, 1),
											padding="same",
											use_bias=False, name = f"{self.block_name}-conv2d_layer_1")
					self.batch_norm_1 = BatchNormalization(name= f"{self.block_name}-batch_norm_layer_1")
					self.activation_1 = Activation("relu", name= f"{self.block_name}-activation_layer_1")
					self.conv2d_2 = Conv2D(filters=self.num_filters,
											kernel_size=3,
											strides=(1, 1),
										padding="same",
											use_bias=False, name = f"{self.block_name}-conv2d_layer_2")
					self.batch_norm_2 = BatchNormalization(name= f"{self.block_name}-batch_norm_layer_2")
					self.activation_2 = Activation("relu", name= f"{self.block_name}-activation_layer_2")

				def call(self, input:tf.Tensor, training:bool=True) -> tf.Tensor:
					'''
					Calling the instance of the `Conv_Block` class will return the layer.

					Parameters
					----------
					input: tf.Tensor
						The input tensor to the layer.

					training: bool
						Whether or not the model is in training mode.

					Returns
					-------
					tf.Tensor

					Raises
					------
					None
					'''
					layer = self.conv2d_1(input)
					layer = self.activation_1(layer)
					layer = self.batch_norm_1(layer, training=training)
					layer = self.conv2d_2(layer)
					layer = self.activation_2(layer)
					layer = self.batch_norm_2(layer, training=training)
					return layer

			class Encoder_Block(tf.keras.Model):
				'''
				This class is responsible for creating a downsampling block of the encoder path which is made of:
				
				1. convolutional block
				2. max pooling
				
				'''
				def __init__(self, num_filters:int, block_name:str):

					super(Encoder_Block, self).__init__()
					self.conv_block = Conv_Block(num_filters, block_name)
					self.block_name = block_name
					self.max_pool = MaxPool2D((2, 2), strides=(2, 2), name=f"{self.block_name}-maxpool")
					
				def call(self, input:tf.Tensor, training=True)-> tuple:
					
					'''
					Calling the instance of the Conv_Block class will return the layer.

					Parameters
					----------
					input: tf.Tensor
						The input tensor to the layer.

					training: bool
						Whether or not the model is in training mode.

					Returns
					-------
					tuple(tf.Tensor, tf.Tensor)

					Raises
					------
					None
					'''

					encoder = self.conv_block(input, training=training)
					encoder_pool = self.max_pool(encoder)
				
					return encoder, encoder_pool

			class ConvTranspose_Block(tf.keras.Model):
				'''
				This class is responsible for creating a deconvolutional block of the upsampling path which is made of:

				1. conv_transpose
				2. activation
				3. batch normalization
				4. conv_transpose
				5. activation
				6. batch normalization

				'''
				
				def __init__(self, num_filters:int, block_name:str):
					super(ConvTranspose_Block, self).__init__()
					self.block_name = block_name
					self.num_filters = num_filters
					self.convT_1 = Conv2DTranspose(filters=self.num_filters,
													kernel_size=3,
													strides=(1, 1), 
													padding="same",
											name = f"{self.block_name}-conv2d_transpose_layer_1")
					
					self.batch_norm_1 = BatchNormalization(name= f"{self.block_name}-batch_norm_layer_1")
					self.activation_1 = Activation("relu", name= f"{self.block_name}-activation_layer_1")
					self.convT_2 = Conv2DTranspose(filters=self.num_filters,
													kernel_size=3,
													padding="same",
													strides=(1, 1),
											name = f"{self.block_name}-conv2d_transpose_layer_2")
					self.batch_norm_2 = BatchNormalization(name= f"{self.block_name}-batch_norm_layer_2")
					self.activation_2 = Activation("relu", name= f"{self.block_name}-activation_layer_2")
				
				def call(self, input, training:bool=True):
										
					'''
					Calling the instance of the `ConvTranspose_Block` class will return the layer.

					Parameters
					----------
					input: tf.Tensor
						The input tensor to the layer.

					training: bool
						Whether or not the model is in training mode.

					Returns
					-------
					tf.Tensor

					Raises
					------
					None
					'''

					layer = self.convT_1(input)
					layer = self.activation_1(layer)
					layer = self.batch_norm_1(layer, training=training)

					layer = self.convT_2(layer)
					layer = self.activation_2(layer)
					layer = self.batch_norm_2(layer, training=training)
					
					return layer

			class Decoder_Block(tf.keras.Model):
				'''
				This class is responsible for creating a downsampling block of the decoder path which is made of:
				
				1. conv_transpose (for upsampling)
				2. activation
				3. batch normalization
				4. dropout
				5. concatenation with the encoder path
				6. deconvolutional block (for performing conv_tranpose operations)
				
				'''
				def __init__(self, num_filters:int, block_name:str):
					
					super(Decoder_Block, self).__init__()
					self.block_name = block_name
					self.num_filters = num_filters
					self.convT = Conv2DTranspose(num_filters,
													kernel_size=5,
													strides=(2, 2),
													padding="same",
										name = f"{self.block_name}-upsample_conv2d_transpose_layer_1")
					self.batch_norm = BatchNormalization(name = f"{self.block_name}-upsample_batchnorm_layer_1")
					self.activation = Activation("relu", name = f"{self.block_name}-upsample_activation_layer_1")
					self.dropout = Dropout(0.4, name = f"{self.block_name}-upsample_dropout_layer_1")
					self.concat = Concatenate(name = f"{self.block_name}-upsample_concatenate_layer_1", axis=-1)
					self.convT_block = ConvTranspose_Block(self.num_filters, self.block_name)
				
				def call(self, input, concat_layer, training=True):
					layer = self.convT(input)
					layer = self.activation(layer)
					layer = self.batch_norm(layer, training=training)
					layer = self.dropout(layer, training=training)
					# concatenate
					layer = self.concat([layer, concat_layer])
					
					# just two consecutive conv_transpose
					layer = self.convT_block(layer, training=training)
					return layer

			class Unet(tf.keras.Model):
				'''
				This class is responsible for creating the UNET model using the encoder and decoder blocks.
				The UNET model is made of:

				1- input layer
				2- four encoder blocks
				3- middle convolutional block
				4- four decoder blocks
				5- final 1x1 convolutional block (for setting the number of output channels)

				'''
				def __init__(self, input_shape:tuple=INPUT_SHAPE):
					super(Unet, self).__init__()

					self.down1 = Encoder_Block(32, "encode_block_1")
					self.down2 = Encoder_Block(64, "encode_block_2")
					self.down3 = Encoder_Block(128, "encode_block_3")
					self.down4 = Encoder_Block(256, "encode_block_4")

					self.center = Conv_Block(512, "center_block_5")

					self.up1 = Decoder_Block(256, "decode_block_1")
					self.up2 = Decoder_Block(128, "decode_block_2")
					self.up3 = Decoder_Block(64, "decode_block_3")
					self.up4 = Decoder_Block(32, "decode_block_4")

					self.last = Conv2D(2, 1, padding="same", activation="relu")
			
				def call(self, input, training:bool) -> tf.Tensor:
					'''
					Calling the Unet instance will create the network layers.

					Parameters
					----------
					input: tf.Tensor
						The input spectrogram tensor to the network.

					training: bool
						Whether or not the model is in training mode.
					
					Returns
					-------
					tf.Tensor - the output tensor of the network after passing through the downsampling and upsampling network layers.

					Raises
					------
					None
					'''
					pad_to_left, pad_to_right, pad_to_top, pad_to_bottom = 0, 0, 0, 0

					'''
					the shapes of the unet should be divisible by 2^n where n is the number of poolings
					therefore we have to pad the input to be divisible by 16 (as we have 4 pooling layers)
					at the end, we also need to crop the image based on the padding we added
					source: https://www.reddit.com/r/deeplearning/comments/t6hqri/comment/hzb7dqo/?utm_source=share&utm_medium=web2x&context=3
					'''
					#padding the height
					if input.shape[1] % 16 != 0:
						pad_to_add = 16 - input.shape[1] % 16
						#the following block tries to pad the data symmetrically and not only to one side             
						if pad_to_add == 1:
							input = ZeroPadding2D(((pad_to_add, 0), (0, 0)))(input)
						else:
							pad_to_top = pad_to_add // 2
							pad_to_bottom = pad_to_add - pad_to_top
							input = ZeroPadding2D(((pad_to_top, pad_to_bottom), (0, 0)))(input)
					
					#padding the width
					if input.shape[2] % 16 != 0:
						pad_to_add = 16 - input.shape[2] % 16
						if pad_to_add == 1:
							input = ZeroPadding2D(((0, 0), (pad_to_add, 0)))(input)
						else:
							pad_to_left = pad_to_add // 2
							pad_to_right = pad_to_add - pad_to_left
							input = ZeroPadding2D(((0, 0), (pad_to_left, pad_to_right)))(input)

					cropping_values = ((pad_to_top, pad_to_bottom), (pad_to_left, pad_to_right)) #storing the padding values for later to crop the output to the original size

					#Encoder Block
					encoder_1, encoder_1_pool = self.down1(input, training=training)
					encoder_2, encoder_2_pool = self.down2(encoder_1_pool, training=training)
					encoder_3, encoder_3_pool = self.down3(encoder_2_pool, training=training)
					encoder_4, encoder_4_pool = self.down4(encoder_3_pool, training=training)

					#Center Convolutional Block
					center = self.center(encoder_4_pool, training=training)

					#Decoder Block
					decoder_5 = self.up1(center, encoder_4, training=training)
					decoder_6 = self.up2(decoder_5, encoder_3, training=training)
					decoder_7 = self.up3(decoder_6, encoder_2, training=training)
					decoder_8 = self.up4(decoder_7, encoder_1, training=training)

					#Final 1x1 Convolutional Block
					last_layer = self.last(decoder_8)

					#last layer to remove the paddings from the output and resizing it to the original size (input size)
					last_layer = Cropping2D(cropping_values)(last_layer)
					return last_layer

			inputs = Input(shape=INPUT_SHAPE, dtype=tf.float32)
			model = Unet()
			model.build(inputs.shape) #using inputs.shape instead of input_shape because inputs.shape also has the additional batch size shape which is set to None
			return model

		print("Creating the model architecture!")
		model = build_Unet()

		return model
	
