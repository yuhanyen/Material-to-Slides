import os
import gradio as gr
from IPython.display import display, Markdown, HTML
import base64
import io
import google.generativeai as genai
import time
from PIL import Image  # Import PIL for image handling

# =====================
# Configure Generative AI with Environment Variable
# =====================
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("環境變數 GOOGLE_API_KEY 未設定，請先設定 API Key")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"無法配置 Google API 金鑰：{e}")
    # 可視需求選擇中斷或提供替代方案
    # raise e

# =====================
# Step 1: Display Uploaded File
# =====================
def display_uploaded_file(file):
    if file is None:
        return gr.Textbox(visible=False), gr.Image(visible=False), gr.Video(visible=False), gr.Audio(visible=False)

    file_path = file.name
    file_type = file_path.split('.')[-1].lower()

    if file_type in ['jpg', 'jpeg', 'png', 'gif']:
        return gr.Textbox(visible=False), gr.Image(value=file_path, label="上傳的圖片", visible=True), gr.Video(visible=False), gr.Audio(visible=False)
    elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
        return gr.Textbox(visible=False), gr.Image(visible=False), gr.Video(value=file_path, label="上傳的影片", visible=True), gr.Audio(visible=False)
    elif file_type in ['mp3', 'wav', 'ogg']:
        return gr.Textbox(visible=False), gr.Image(visible=False), gr.Video(visible=False), gr.Audio(value=file_path, label="上傳的音訊", visible=True)
    elif file_type in ['txt']:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return gr.Textbox(value=content, label="上傳的文字檔案內容", visible=True), gr.Image(visible=False), gr.Video(visible=False), gr.Audio(visible=False)
    else:
        return gr.Textbox(value="不支援的檔案格式", label="顯示內容", visible=True), gr.Image(visible=False), gr.Video(visible=False), gr.Audio(visible=False)

# =====================
# Step 2: Process with Gemini
# =====================
def process_with_gemini(file, prompt_text):
    if file is None:
        return "請先上傳檔案。"
    if not prompt_text:
        return "請輸入提示詞。"

    file_path = file.name
    file_type = file_path.split('.')[-1].lower()

    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")

        if file_type in ['jpg', 'jpeg', 'png', 'gif']:
            img = Image.open(file_path)
            response = model.generate_content([img, prompt_text])
            return response.text

        elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
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
            genai.delete_file(video_file.name)
            return response.text

        elif file_type in ['txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            response = model.generate_content([content, prompt_text])
            return response.text

        elif file_type in ['mp3', 'wav', 'ogg']:
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
            genai.delete_file(audio_file.name)
            return response.text

        else:
            return "不支援的檔案格式，無法傳送至 Gemini API"

    except Exception as e:
        return f"呼叫 Gemini API 時發生錯誤：{e}"

# =====================
# Step 3: Copy Gemini Output to Editor
# =====================
def copy_response_to_editor(gemini_response_text):
    return gr.Textbox(value=gemini_response_text, visible=True)

# =====================
# Step 4: Generate Transcript & Presentation
# =====================
def generate_transcript_and_presentation(text, prompt_text_step4):
    prompt = f"{prompt_text_step4}，內容: {text}。"
    return Gemini_GenText(prompt)

# Gemini Text Generation Helper
def Gemini_GenText(prompt):
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        return "錯誤：環境變數 GOOGLE_API_KEY 未設定"
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text

# =====================
# Gradio UI
# =====================
with gr.Blocks() as demo:
    gr.Markdown("## 多模態語言模型範例")

    # Step 1: File Upload
    gr.Markdown("### Step1: 檔案上傳")
    file_input = gr.File(label="請上傳檔案 (圖片、影片、音訊或文字檔)")

    output_text = gr.Textbox(label="上傳的文字檔案內容", visible=False)
    output_image = gr.Image(label="上傳的圖片", visible=False)
    output_video = gr.Video(label="上傳的影片", visible=False)
    output_audio = gr.Audio(label="上傳的音訊", visible=False)

    file_input.change(
        fn=display_uploaded_file,
        inputs=file_input,
        outputs=[output_text, output_image, output_video, output_audio]
    )

    # Step 2: Multimodal Inference
    gr.Markdown("### Step2: 多模態推論")
    prompt_input = gr.Textbox(label="請輸入提示詞(可修改)", value="請幫我解譯上傳內容，並生成重點300字")
    submit_button = gr.Button("送出給 Gemini API")
    gemini_response = gr.Textbox(label="Gemini API 回應")

    submit_button.click(
        fn=process_with_gemini,
        inputs=[file_input, prompt_input],
        outputs=gemini_response
    )

    # Step 3: Edit Text
    gr.Markdown("### Step3: 文案編輯")
    edit_button = gr.Button(value="進入文字編輯模式")
    edited_text = gr.Textbox(label="編輯文字")
    edit_button.click(
        fn=copy_response_to_editor,
        inputs=gemini_response,
        outputs=edited_text
    )

    # Step 4: Presentation Generation
    gr.Markdown("### Step4: 推論生成簡報(含逐字稿)")
    prompt_input_step4 = gr.Textbox(label="請輸入生成簡報提示詞(可修改)", value="我需要製作簡報一頁，所以請重點摘要包含簡報標題、三個重點標題及細節")
    generate_button_step4 = gr.Button(value="生成簡報")
    output_presentation = gr.Textbox(label="生成簡報Markdown")

    generate_button_step4.click(
        fn=generate_transcript_and_presentation,
        inputs=[edited_text, prompt_input_step4],
        outputs=[output_presentation]
    )

# =====================
# Launch App
# =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
