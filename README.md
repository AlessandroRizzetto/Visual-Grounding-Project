# Visual Grounding on RefCOCOg with YOLO & CLIP

A visual grounding experiment for the Deep Learning course: given an image and a caption, we locate the correct region using two approaches:

1. **Baseline**  
   - Detect objects with YOLO (Ultralytics).  
   - Compute similarity between CLIP (RN50) vectors of each crop and the text.  
2. **ExtendedClip**  
   - Extend CLIP with a small trainable encoder on text and image features.  
   - Train/fine-tune on RefCOCOg to improve IoU and cosine similarity.

## 📂 Structure

- **Environment Setup** – install dependencies, imports, paths.  
- **Dataset & Metrics** – custom dataset class for RefCOCOg + definitions of IoU, mIoU, cosine similarity.  
- **Baseline** – YOLO+CLIP zero-shot.  
- **ExtendedClip** – architecture with additional encoders, training loop, custom loss.  
- **Training & Evaluation** – plots of loss and accuracy, metric comparison.  
- **Results** – table of mIoU and cosine similarity.

## 🚀 Running on Colab

1. Go to **File → Save a copy in Drive** to create your own Colab copy.  
2. Ensure GPU runtime: **Runtime → Change runtime type → GPU**.  
3. Run each cell in order (Shift + Enter) to install, prepare data, train and evaluate.

## 🔧 Requirements (installed automatically on Colab)

```bash
!pip install torch torchvision clip ultralytics pycocotools textdistance matplotlib pillow numpy
