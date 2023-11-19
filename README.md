---
title: DrugDOpenV
emoji: 🐨
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.28.2
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


To run the provided Python script using Streamlit, follow these steps:

### 1. Set Up Your Environment
Clone the repo

Create anaconda environment enable
run the command
pip install -r requirements.txt
```


Note: RDKit might require a specific installation process, especially on certain operating systems. Ensure you follow the official RDKit installation guidelines.

### 2. Prepare Your Script

`.
- Ensure that the model file (`lipophilicity_openvino.xml`) is accessible at the specified path in your script or update the `model_path` variable to the correct path where your model is stored.

### 3. Run the Streamlit App

- Open a terminal or command prompt.
- Navigate to the directory where your script is located.
- Run the script using Streamlit. For example, if your file is named `lipophilicity_app.py`, you would run:

```bash
streamlit run app.py
```

- Streamlit will start a local web server and provide you with a URL (usually `http://localhost:8501`).

### 4. Interact with the App

- Open the provided URL in a web browser.
- You'll see the Streamlit interface with the title "Lipophilicity Prediction App".
- Use the dropdown menu to select a SMILES string.
- Click the 'Predict Lipophilicity' button to see the prediction and the visual representation of the molecule.

### Additional Considerations

- Ensure that your environment supports all the necessary libraries and dependencies. For example, RDKit and OpenVINO have specific requirements for installation.
- If you plan to share this app or deploy it, consider using Streamlit sharing or another cloud service that supports Python and the required libraries.
- Be mindful of the computational resources required for running the model, especially if deploying this app for broader use.

Running a Streamlit app is generally straightforward and provides a powerful way to create interactive data appl
