import gradio as gr
import pandas as pd
import requests
from io import StringIO

# Define a function that handles the API request
def send_request(file):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file.name)

    # Convert DataFrame to JSON (this depends on your API requirements)
    data_json = df.to_json(orient='split')
    payload = {'data': data_json}

    # Example API endpoint (replace with your actual endpoint)
    url = "http://api:8000/best_model"

    # Send POST request to the API (you may need to adjust headers and data format)
    response = requests.post(url, json=payload)

    res = pd.read_json(response.json()['pred'], orient='split')
    res.columns = ['Prediction']
    pred_res = pd.concat([df, res], axis=1)

    output_file = "processed_file.csv"
    pred_res.to_csv(output_file, index=False)

    return pred_res, output_file

    # csv_buffer = StringIO()
    # pred_res.to_csv(csv_buffer, index=False)
    # csv_buffer.seek(0)
    # return gr.File(content=csv_buffer.getvalue(), name="response.csv")


# Create a Gradio interface
with gr.Blocks() as demo:
    # Add a file uploader for CSV files
    csv_file = gr.File(label="Upload your CSV file")

    # Add a button to trigger the API request
    submit_button = gr.Button("Predict")
    download_button = gr.File(label="Download CSV", interactive=False)
    # Define the action when the button is clicked

    submit_button.click(fn=send_request, inputs=csv_file, outputs=[
        gr.DataFrame(label="Processed DataFrame"),  # Output 1: Show DataFrame
        gr.File(label="Download Processed CSV")  # Output 2: File download
    ])

# Launch the interface
demo.launch()