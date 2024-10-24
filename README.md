# 2D to 3D Image Generation Project

This project converts a 2D image into a 3D representation using depth generation techniques, 3D mesh reconstruction, and AI-based enhancements for missing parts of the image.

## Dependencies

Ensure you have Python 3.6 or later installed. You can install the necessary dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Project Setup and Execution

1. **Clone the repository and navigate to the project directory:**

```bash
git clone <REPOSITORY_URL>
cd 2d_to_3d
```

2. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the `main.py` file to start the application:**

```bash
python main.py
```

4. **Access the web interface in your browser:**

```
http://127.0.0.1:5000
```

## Web Interface Usage

### Step 1: Upload an Image

1. Select an image from your computer using the "Choose File" button.
2. Click "Upload and Process" to start the transformation.

### Step 2: View the Original 3D Image

After the upload and processing, the interface will display the generated 3D image. You can validate this step and move on by clicking "Validate and Proceed."

### Step 3: View the AI-enhanced 3D Image

The AI will generate missing parts of the image, improving the 3D model. This updated image will be shown. You can validate and proceed as before.

### Step 4: View the Final 3D Image

The final 3D model will be displayed with all enhancements applied.

## Example Results

### Step 1: Original Image

![Original Image](static/img/etape_1.png)

### Step 2: 3D Image Generated

![3D Image](static/img/etape_2.png)

### Step 3: AI-enhanced 3D Image

![AI-enhanced 3D Image](static/img/etape_3.png)

### Step 4: Final 3D Image

![Final 3D Image](static/img/etape_4.png)

## Technical Details

- **Depth Estimation**: The project uses the MiDaS model to estimate depth from a 2D image.
- **Depth to Point Cloud**: Converts the depth map into a 3D point cloud.
- **Point Cloud to Mesh**: Converts the point cloud into a 3D mesh model.
- **AI-based Enhancements**:
  - **Inpainting for Missing Parts**: Completes missing sections using AI-based inpainting.
  - **Texture Generation**: GPT-4 analyzes the image and improves textures based on AI suggestions.
  - **Annotation & Segmentation**: Assists in segmenting objects within the image for better 3D reconstruction.
  - **Parameter Optimization**: GPT-4 optimizes reconstruction parameters based on analysis.
  - **3D Model Synthesis**: GPT-4 helps fuse data to create more precise 3D models.
