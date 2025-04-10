# Professional Image Converter Application

This is a Python-based graphical image editor that lets users convert and enhance images using a variety of artistic effects. The application features a modern, professional user interface with standard window controls (minimize, maximize, close) and dynamic, resizable layouts. It is designed to deliver a seamless image editing experience with real-time effect adjustments.

## Features

- **Standard Window Controls & Resizable UI**  
  The application window includes the typical OS-provided controls (minimize, maximize, close) and can be resized dynamically while updating the layout accordingly.

- **Image Conversion & Artistic Effects**  
  Choose from a range of creative effects like pencil sketch, color sketch, stylization, cartoon, vintage, HDR, glitch, and more. Effects can be applied with adjustable intensity.

- **Real-time Image Adjustments**  
  Fine-tune the processed image with sliders for effect intensity, brightness, contrast, and saturation.

- **Seamless File Dialog Integration**  
  Utilizes Tkinter dialogs that remain topmost, ensuring the main application does not minimize unexpectedly when selecting or saving images.

- **Centered Image Container**  
  The loaded image is displayed in a central container that preserves its aspect ratio, giving your application a professional look.

## Installation

1. **Clone the Repository:**


Navigate to the Project Directory:

bash
Copy
Edit
cd image-converter
Set Up a Virtual Environment (Optional but Recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
Install Required Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
The requirements.txt should include at least the following packages:

nginx
Copy
Edit
pygame
opencv-python
pillow
Usage
Run the Application:

bash
Copy
Edit
python app.py
(Replace app.py with your main application file if it differs.)

How to Use:

Load an Image: Click the "Convert Image" button to open a file dialog, select your image, and load it into the application. The image will be resized (while maintaining its aspect ratio) to fit within a centered container.

Apply Effects: Use the "Prev Effect" and "Next Effect" buttons to cycle through different artistic effects. Adjust the intensity of the chosen effect, as well as brightness, contrast, and saturation, using the provided sliders.

Save Your Work: Once you are happy with the edits, click the "Save Image" button to open a file dialog for saving your modified image.

Window Controls:

The window features standard OS controls (minimize, maximize, close) so you can manage the application as you would any professional desktop program.

The UI dynamically adapts to window resizing.

Contributing
Contributions are welcome! If you have ideas for additional effects, UI improvements, or bug fixes, please feel free to submit a pull request or open an issue.