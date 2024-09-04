import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from groq import Groq
#kfkjfn
# Azure Computer Vision API Configuration
AZURE_API_KEY = 'e2ef85d27fa74e7ea7d2a1b6b608adaf'
AZURE_ENDPOINT = 'https://bharaths.cognitiveservices.azure.com/vision/v3.2/analyze'

# Groq API Configuration
CHATBOT_API_KEY = "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"
client = Groq(api_key=CHATBOT_API_KEY)


def send_to_azure_api(image):
    api_url = AZURE_ENDPOINT
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_API_KEY,
        'Content-Type': 'application/octet-stream'
    }
    params = {
        'visualFeatures': 'Tags,Description,Objects'
    }

    try:
        # Convert image to bytes
        with BytesIO() as buffer:
            image.save(buffer, format=image.format)
            image_data = buffer.getvalue()

        response = requests.post(api_url, headers=headers, params=params, data=image_data)
        if response.status_code == 200:
            return response.json()
        else:
            error_message = f"Error: API request failed with status code {response.status_code}"
            return {'error': error_message, 'details': response.text}
    except Exception as e:
        return {'error': str(e), 'details': str(e)}


def extract_caption(api_results):
    try:
        captions = api_results.get('description', {}).get('captions', [])
        if captions:
            return captions[0].get('text', 'No caption available')
        return 'No caption available'
    except Exception as e:
        return 'Error extracting caption'


def send_to_chatbot(caption):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{'role': 'user',
                       'content': caption + " my location is Erode Tamil Nadu pincode 637303. Give nearby shops"}],
            model='llama3-8b-8192',
            max_tokens=1000
        )
        if chat_completion.choices and chat_completion.choices[0].message:
            return chat_completion.choices[0].message.content
        else:
            return 'No valid response from chatbot'
    except Exception as e:
        return f"Exception occurred while contacting chatbot: {e}"


# Streamlit UI
st.title('Image Analysis and Chatbot Interaction')

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg', 'gif'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Send request to Azure API
    api_results = send_to_azure_api(image)

    # Extract caption from API results
    caption = extract_caption(api_results)
    st.write(f"Caption: {caption}")

    # Send caption to the LLM chatbot
    chat_response = send_to_chatbot(caption)
    st.write(f"Chatbot Response: {chat_response}")
