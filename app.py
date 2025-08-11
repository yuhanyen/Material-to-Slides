import os
import time
import base64
import io

import gradio as gr
import google.generativeai as genai
from PIL import Image  # 圖片處理

# =====================
# Configure Generative AI (from ENV)
# =====================
def _configure_genai_or_raise():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("環境變數 GOOGLE_API_KEY 未設定，請在 Render 或執行環境加入該變數。")
    genai.configure(api_key=api_key)

try:
    _configure_genai_or_raise()
except Exception as e:
    # 在 Render 的啟動日誌可看到這段訊息，利於除錯
    print(f"[Startup] 無法配置 Google API 金鑰：{e}")

# =====================
# Step 1: 檔案顯示
# =====================
def display_uploaded_file(file):
    if file is None:
        return (
            gr.Textbox(visible=False),
            gr.Image(visible=False),
            gr.Video(visible=False),
            gr.Audio(visible=False),
        )

    file_path = file.name
    file_type = (file_path.split(".")[-1] or "").lower()

    if file_type in ["jpg", "jpeg", "png", "gif"]:
        return (
            gr.Textbox(visible=False),
            gr.Image(value=file_path, label="上傳的圖片", visible=True),
            gr.Video(visible=False),
            gr.Audio(visible=False),
        )
    elif file_type in ["mp4", "avi", "mov", "mkv"]:
        return (
            gr.Textbox(visible=False),
            gr.Image(visible=False),
            gr.Video(value=file_path, label="上傳的影片", visible=True),
            gr.Audio(visible=False),
        )
    elif file_type in ["mp3", "wav", "ogg"]:
        return (
            gr.Textbox(visible=False),
            gr.Image(visible=False),
            gr.Video(visible=False),
            gr.Audio(value=file_path, label="上傳的音訊", visible=True),
        )
    elif file_type in ["txt"]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return (
            gr.Textbox(value=content, label="上傳的文字檔案內容", visible=True),
            gr.Image(visible=False),
            gr.Video(visible=False),
            gr.Audio(visible=False),
        )
    else:
        return (
            gr.Textbox(value="不支援的檔案格式", label="顯示內容", visible=True),
            gr.Image(visible=False),
            gr.Video(visible=False),
            gr.Audio(visible=False),
        )

# =====================
# Step 2: Gemini 多模態推論
# =====================
def process_with_gemini(file, prompt_text):
    if file is None:
        return "請先上傳檔案。"
    if not prompt_text:
        return "請輸入提示詞。"

    _configure_genai_or_raise()
    file_path = file.name
    file_type = (file_path.split(".")[-1] or "").lower()

    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")

        if file_type in ["jpg", "jpeg", "png", "gif"]:
            img = Image.open(file_path)
            response = model.generate_content([img, prompt_text])
            return getattr(response, "text", "(無文字回覆)")

        elif file_type in ["mp4", "avi", "mov", "mkv"]:
            print("[Gemini] Uploading video...")
            video_file = genai.upload_file(path=file_path)
            print(f"[Gemini] Completed upload: {getattr(video_file, 'uri', '')}")

            while getattr(video_file, "state", None) and video_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(10)
                video_file = genai.get_file(video_file.name)

            if getattr(video_file, "state", None) and video_file.state.name == "FAILED":
                return f"影片上傳失敗: {video_file.state.name}"

            response = model.generate_content([video_file, prompt_text], request_options={"timeout": 600})
            # 清理雲端暫存檔
            try:
                genai.delete_file(video_file.name)
            except Exception:
                pass
            return getattr(response, "text", "(無文字回覆)")

        elif file_type in ["txt"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            response = model.generate_content([content, prompt_text])
            return getattr(response, "text", "(無文字回覆)")

        elif file_type in ["mp3", "wav", "ogg"]:
            print("[Gemini] Uploading audio...")
            audio_file = genai.upload_file(path=file_path)
            print(f"[Gemini] Completed upload: {getattr(audio_file, 'uri', '')}")

            while getattr(audio_file, "state", None) and audio_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(10)
                audio_file = genai.get_file(audio_file.name)

            if getattr(audio_file, "state", None) and audio_file.state.name == "FAILED":
                return f"音檔上傳失敗: {audio_file.state.name}"

            response = model.generate_content([audio_file, prompt_text], request_options={"timeout": 600})
            # 清理雲端暫存檔
            try:
                genai.delete_file(audio_file.name)
            except Exception:
                pass
            return getattr(response, "text", "(無文字回覆)")

        else:
            return "不支援的檔案格式，無法傳送至 Gemini API"

    except Exception as e:
        return f"呼叫 Gemini API 時發生錯誤：{e}"

# =====================
# Step 3: 把回應放進編輯器
# =====================
def copy_response_to_editor(gemini_response_text):
    return gr.Textbox(value=gemini_response_text or "", visible=True)

# =====================
# Step 4: 逐字稿 + 簡報生成
# =====================
def Gemini_GenText(prompt: str) -> str:
    try:
        _configure_genai_or_raise()
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        return getattr(response, "text", "(無文字回覆)")
    except Exception as e:
        return f"呼叫 Gemini API 時發生錯誤：{e}"

def generate_transcript_and_presentation(text, prompt_text_step4):
    prompt = f"{prompt_text_step4}，內容: {text}。"
    return Gemini_GenText(prompt)

# =====================
# Gradio 介面
# =====================
with gr.Blocks() as demo:
    gr.Markdown("## 多模態語言模型範例")

    # Step 1: 檔案上傳
    gr.Markdown("### Step1: 檔案上傳")
    file_input = gr.File(label="請上傳檔案 (圖片、影片、音訊或文字檔)")

    output_text = gr.Textbox(label="上傳的文字檔案內容", visible=False)
    output_image = gr.Image(label="上傳的圖片", visible=False)
    output_video = gr.Video(label="上傳的影片", visible=False)
    output_audio = gr.Audio(label="上傳的音訊", visible=False)

    file_input.change(
        fn=display_uploaded_file,
        inputs=file_input,
        outputs=[output_text, output_image, output_video, output_audio],
    )

    # Step 2: 多模態推論
    gr.Markdown("### Step2: 多模態推論")
    prompt_input = gr.Textbox(label="請輸入提示詞(可修改)", value="請幫我解譯上傳內容，並生成重點300字")
    submit_button = gr.Button("送出給 Gemini API")
    gemini_response = gr.Textbox(label="Gemini API 回應")

    submit_button.click(
        fn=process_with_gemini,
        inputs=[file_input, prompt_input],
        outputs=gemini_response,
    )

    # Step 3: 文案編輯
    gr.Markdown("### Step3: 文案編輯")
    edit_button = gr.Button(value="進入文字編輯模式")
    edited_text = gr.Textbox(label="編輯文字")
    edit_button.click(
        fn=copy_response_to_editor,
        inputs=gemini_response,
        outputs=edited_text,
    )

    # Step 4: 簡報生成
    gr.Markdown("### Step4: 推論生成簡報(含逐字稿)")
    prompt_input_step4 = gr.Textbox(
        label="請輸入生成簡報提示詞(可修改)",
        value="我需要製作簡報一頁，所以請重點摘要包含簡報標題、三個重點標題及細節",
    )
    generate_button_step4 = gr.Button(value="生成簡報")
    output_presentation = gr.Textbox(label="生成簡報Markdown")

    generate_button_step4.click(
        fn=generate_transcript_and_presentation,
        inputs=[edited_text, prompt_input_step4],
        outputs=[output_presentation],
    )

# =====================
# 啟動
# =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    # Render 需綁在 0.0.0.0，且使用平台提供的 PORT
    demo.launch(server_name="0.0.0.0", server_port=port)
