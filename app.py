import gradio as gr
from IPython.display import display, Markdown, HTML
import base64
import io
import google.generativeai as genai
from google.colab import userdata
import time
from PIL import Image # Import PIL for image handling

# Configure Generative AI
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"無法配置 Google API 金鑰：{e}")
    # Consider adding a fallback or raising an error if the API key is essential

def display_uploaded_file(file):
    if file is None:
        # When no file is uploaded, hide all output components
        return gr.Textbox(visible=False), gr.Image(visible=False), gr.Video(visible=False), gr.Audio(visible=False)

    file_path = file.name
    file_type = file_path.split('.')[-1].lower()

    if file_type in ['jpg', 'jpeg', 'png', 'gif']:
        # If it's an image, return the image component with the file path and hide others
        return gr.Textbox(visible=False), gr.Image(value=file_path, label="上傳的圖片", visible=True), gr.Video(visible=False), gr.Audio(visible=False)
    elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
        # If it's a video, return the video component with the file path and hide others
        return gr.Textbox(visible=False), gr.Image(visible=False), gr.Video(value=file_path, label="上傳的影片", visible=True), gr.Audio(visible=False)
    elif file_type in ['mp3', 'wav', 'ogg']:
        # If it's an audio file, return the audio component with the file path and hide others
        return gr.Textbox(visible=False), gr.Image(visible=False), gr.Video(visible=False), gr.Audio(visible=False)
    elif file_type in ['txt']:
        # If it's a text file, read the content and return a textbox with the content, hide others
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return gr.Textbox(value=content, label="上傳的文字檔案內容", visible=True), gr.Image(visible=False), gr.Video(visible=False), gr.Audio(visible=False)
    else:
        # If the file type is not supported, display an error message in a textbox and hide others
        return gr.Textbox(value="不支援的檔案格式", label="顯示內容", visible=True), gr.Image(visible=False), gr.Video(visible=False), gr.Audio(visible=False)

def process_with_gemini(file, prompt_text):
    if file is None:
        return "請先上傳檔案。"
    if not prompt_text:
        return "請輸入提示詞。"

    file_path = file.name
    file_type = file_path.split('.')[-1].lower()

    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-pro") # Choose an appropriate model

        if file_type in ['jpg', 'jpeg', 'png', 'gif']:
            # Handle image with prompt
            img = Image.open(file_path) # Open the image file
            response = model.generate_content([img, prompt_text]) # Pass the PIL Image object
            return response.text
        elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
            # Handle video with prompt - requires uploading to GenAI first
            print(f"Uploading file...")
            video_file = genai.upload_file(path=file_path)
            print(f"Completed upload: {video_file.uri}")

            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(10)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
              return f"影片上傳失敗: {video_file.state.name}"

            response = model.generate_content([video_file, prompt_text], request_options={"timeout": 600})
            genai.delete_file(video_file.name) # Clean up
            return response.text

        elif file_type in ['txt']:
            # Handle text with prompt
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            response = model.generate_content([content, prompt_text])
            return response.text
        elif file_type in ['mp3', 'wav', 'ogg']:
             # Handle audio with prompt - requires uploading to GenAI first
            print(f"Uploading file...")
            audio_file = genai.upload_file(path=file_path)
            print(f"Completed upload: {audio_file.uri}")

            while audio_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(10)
                audio_file = genai.get_file(audio_file.name)

            if audio_file.state.name == "FAILED":
              return f"音檔上傳失敗: {audio_file.state.name}"

            response = model.generate_content([audio_file, prompt_text], request_options={"timeout": 600})
            genai.delete_file(audio_file.name) # Clean up
            return response.text


        else:
            return "不支援的檔案格式，無法傳送至 Gemini API"

    except Exception as e:
        return f"呼叫 Gemini API 時發生錯誤：{e}"

def copy_response_to_editor(gemini_response_text):
    return gr.Textbox(value=gemini_response_text, visible=True)

# Placeholder function for Step 4 generation
def generate_transcript_and_presentation(text, prompt_text_step4):
    prompt = f"{prompt_text_step4}，內容: {text}。"
    return Gemini_GenText(prompt)

def Gemini_GenText(prompt):
      GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
      genai.configure(api_key=GOOGLE_API_KEY)
      model = genai.GenerativeModel("gemini-2.5-pro")
      response = model.generate_content(prompt)
      return response.text

with gr.Blocks() as demo:
    gr.Markdown("## 多模態語言模型範例")

    gr.Markdown("### Step1: 檔案上傳")
    file_input = gr.File(label="請上傳檔案 (圖片、影片、音訊或文字檔)")

    # Create separate components for displaying different file types
    output_text = gr.Textbox(label="上傳的文字檔案內容", visible=False)
    output_image = gr.Image(label="上傳的圖片", visible=False)
    output_video = gr.Video(label="上傳的影片", visible=False)
    output_audio = gr.Audio(label="上傳的音訊", visible=False)

    file_input.change(
        fn=display_uploaded_file,
        inputs=file_input,
        outputs=[output_text, output_image, output_video, output_audio]
    )

    gr.Markdown("### Step2: 多模態推論") # Modified markdown text
    prompt_input = gr.Textbox(label="請輸入提示詞(可修改)", value="請幫我解譯上傳內容，並生成重點300字")
    submit_button = gr.Button("送出給 Gemini API")

    # Component to display Gemini's response
    gemini_response = gr.Textbox(label="Gemini API 回應")

    submit_button.click(
        fn=process_with_gemini,
        inputs=[file_input, prompt_input],
        outputs=gemini_response
    )

    gr.Markdown("### Step3: 文案編輯") # Add the new markdown for Step 3
    # Add the "進入文字編輯" button and the text editing box
    edit_button = gr.Button(value="進入文字編輯模式")
    edited_text = gr.Textbox(label="編輯文字")

    # Add the click event for the edit button
    edit_button.click(
        fn=copy_response_to_editor,
        inputs=gemini_response,
        outputs=edited_text
    )

    # Add the new markdown heading for Step 4
    gr.Markdown("### Step4: 推論生成簡報(含逐字稿)")
    # Add the prompt input for Step 4
    prompt_input_step4 = gr.Textbox(label="請輸入生成簡報提示詞(可修改)", value="我需要製作簡報一頁，所以請重點摘要包含簡報標題、三個重點標題及細節")
    # Add the button for Step 4
    generate_button_step4 = gr.Button(value="生成簡報") # Define the button
    # Add the output textboxes for Step 4
    output_presentation = gr.Textbox(label="生成簡報Markdown")

    # Connect the Step 4 button to the placeholder function, passing the required inputs
    generate_button_step4.click(
        fn=generate_transcript_and_presentation,
        inputs=[edited_text, prompt_input_step4],
        outputs=[output_presentation]
    )


# Run the interface
if __name__ == "__main__":
    demo = setup_gradio_interface()
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
