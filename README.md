## Public Deployment

A working deployment of the crop and weed counting application has been created using Streamlit Community Cloud for testing and verification.

### Current Test Deployment

[Open the Crop & Weed Counter](https://crop-weed-counter-afridi.streamlit.app/)

The current deployment is hosted from the `afridi-deployment` branch through a personal fork of the team repository. It is intended as a working deployment for testing and verification. The final project deployment can later be moved to the main team repository or redeployed under a neutral project-owned URL.

## Deployment Setup

The deployed application uses:

* `app.py` as the Streamlit entry point
* `requirements.txt` for the minimal dependencies required by the deployed application
* `requirements-dev.txt` for the full local development environment
* `.streamlit/config.toml` for Streamlit configuration
* `runs/colab_50epoch/best.pt` as the trained YOLO model weights

## Running Locally

Install the required application dependencies:

```bash
pip install -r requirements.txt
```

Start the application with:

```bash
python -m streamlit run app.py
```

If Streamlit is available directly in the active environment, it can also be started with:

```bash
streamlit run app.py
```

After startup, open the local URL provided by Streamlit, usually:

```text
http://localhost:8501
```

## Using the Application

1. Open the application.
2. Upload a JPG, JPEG, or PNG plant image.
3. Wait for the YOLO model to process the image.
4. The application displays the image with detected plants annotated.
5. The interface displays:

   * crop count
   * weed count
   * total plant count

## Deployment Verification

The deployed application was tested with both simpler and more crowded plant images.

The following functionality was verified successfully:

* application loads through the public Streamlit deployment
* image upload works
* YOLO model weights load successfully
* model inference completes
* annotated detections are displayed
* crop count is displayed
* weed count is displayed
* total count is displayed

Model accuracy is evaluated separately from deployment functionality. These deployment tests confirm that the application runs and returns predictions successfully, but do not imply that every predicted count is accurate.
