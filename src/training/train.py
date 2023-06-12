import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_model(num_classes):
    # Load the pre-trained ResNet50 model
    base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model's layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def train_model(train_data_dir, validation_data_dir, model_save_path):
    # Define the parameters for training
    num_classes = 20
    batch_size = 32
    epochs = 10
    image_size = (224, 224)

    # Prepare the data augmentation and preprocessing pipeline
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Load the training and validation datasets
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=image_size,
                                                        batch_size=batch_size,
                                                        class_mode="categorical")
    
    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=image_size,
                                                                  batch_size=batch_size,
                                                                  class_mode="categorical")
    
    # Build the model
    model = build_model(num_classes)

    # Compile the model
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
        
    """ Train the model

    Usually, I would use the code below but for demonstration purposes and because of the limited dataset, I will use steps per epoch and validation steps as 1
    steps_per_epoch=train_generator.samples // batch_size
    validation_steps=validation_generator.samples // batch_size

    """
    model.fit(train_generator,steps_per_epoch=1,validation_data=validation_generator,validation_steps=1,epochs=epochs)    

    # Save the trained model
    model.save(model_save_path)


if __name__ == "__main__":
    train_data_dir = r"C:\Users\0xdan\Documents\CS\Catergories\Image and Video Processing\ImageRecognition\BrandLogoPredictor\data"
    validation_data_dir = r"C:\Users\0xdan\Documents\CS\Catergories\Image and Video Processing\ImageRecognition\BrandLogoPredictor\data"
    model_save_path =r"C:\Users\0xdan\Documents\CS\Catergories\Image and Video Processing\ImageRecognition\BrandLogoPredictor\models\model.h5"
    train_model(train_data_dir, validation_data_dir, model_save_path)