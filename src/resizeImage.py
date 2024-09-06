import cv2
import os


def batch_resize_images(input_folder, output_folder, size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                image = cv2.imread(image_path)

                if image is None:
                    raise ValueError(f"Failed to load image {filename}. It may be corrupted.")

                resized_image = cv2.resize(image, size)
                cv2.imwrite(output_path, resized_image)
                print(f"Resized and saved {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
batch_resize_images('DataSet', 'resized_images')
