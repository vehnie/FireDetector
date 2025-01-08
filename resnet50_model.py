import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, 
    GlobalAveragePooling2D, Dense, Activation, Add
)
from tensorflow.keras.models import Model

def identity_block(x, filters, kernel_size=3):

    f1, f2, f3 = filters
    
    # salvar valor de entrada
    shortcut = x
    
    # primeiro componente
    x = Conv2D(f1, kernel_size=1, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # segundo componente
    x = Conv2D(f2, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # terceiro componente
    x = Conv2D(f3, kernel_size=1, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    
    # add shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def conv_block(x, filters, kernel_size=3, stride=2):

    f1, f2, f3 = filters
    
    # salvar valor de entrada
    shortcut = x
    
    # primeiro componente
    x = Conv2D(f1, kernel_size=1, strides=stride)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # segundo componente
    x = Conv2D(f2, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # terceiro componente
    x = Conv2D(f3, kernel_size=1, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)
    
    # shortcut
    shortcut = Conv2D(f3, kernel_size=1, strides=stride)(shortcut)
    shortcut = BatchNormalization()(shortcut)
    
    # adicionar shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def create_resnet50_base(input_shape=(224, 224, 3)):

    # Input layer
    input_tensor = Input(shape=input_shape)
    
    # Convulsao inicial
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # stage 1
    x = conv_block(x, [64, 64, 256], stride=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])
    
    # stage 2
    x = conv_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    
    # stage 3
    x = conv_block(x, [256, 256, 1024])
    for _ in range(5):
        x = identity_block(x, [256, 256, 1024])
    
    # stage 4
    x = conv_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    
    # global average pooling
    x = GlobalAveragePooling2D()(x)
    
    model = Model(inputs=input_tensor, outputs=x, name='resnet50_base')
    return model

def create_fire_detection_model(input_shape=(224, 224, 3)):

    # imagem input
    resnet_base = create_resnet50_base(input_shape)
    
    # temperatura input
    temp_input = Input(shape=(1,), name='temperature_input')
    
    # extrair features da imagem
    img_features = resnet_base.output
    
    # combinar features da imagem com a temperatura
    combined = tf.keras.layers.Concatenate()([img_features, temp_input])
    
    # camada densa adicional
    x = Dense(1024, activation='relu')(combined)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # criar o modelo completo
    model = Model(inputs=[resnet_base.input, temp_input], outputs=output)
    
    return model

def preprocess_data(image, temperature):

    # Resize 224x224
    image = tf.image.resize(image, (224, 224))
    
    # converter para float 32
    image = tf.cast(image, tf.float32) / 255.0
    
    # nromalizar a temperatura
    temperature = tf.cast(temperature, tf.float32)
    temperature = (temperature - 20.0) / (100.0 - 20.0)  # Normalize between min and max temps
    
    return image, temperature

def train_model(model, train_images, train_temps, train_labels, 
                val_images=None, val_temps=None, val_labels=None,
                epochs=10, batch_size=32):

    # compilar o modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # preparar os dados de validação, se fornecidos
    validation_data = None
    if val_images is not None and val_temps is not None and val_labels is not None:
        validation_data = ([val_images, val_temps], val_labels)
    
    # treinar o modelo
    history = model.fit(
        [train_images, train_temps],
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data
    )
    
    return history
