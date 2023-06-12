from utils import load_image, load_model, predict_logo,get_predicted,get_classes



def main():
    # Get all class labels
    logos = get_classes("dataset_folder_path")
    # Define the path to the input image
    image_path ="test_img_folder_path"

    # Load and preprocess the image
    image = load_image(image_path)

    # Load the pre-trained model
    model_path = "model_path"
    model = load_model(model_path)

    # Predict the company logo
    predicted_logo = predict_logo(model, image)

    # Print the predicted company logo
    print("Predicted Company Index:", predicted_logo)
    print("Predicted Company Logo:", get_predicted(logos,predicted_logo))


if __name__ == "__main__":
    main()