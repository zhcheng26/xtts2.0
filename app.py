import os
import shutil

import gradio as gr

from train import train
from infer import infer

MODELS_DIR = "models"
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"

for _d in (MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_available_models() -> list[str]:
    """Return model names that have a .pth file in models/<name>/."""
    models = []
    if not os.path.isdir(MODELS_DIR):
        return models
    for name in sorted(os.listdir(MODELS_DIR)):
        model_dir = os.path.join(MODELS_DIR, name)
        if os.path.isdir(model_dir):
            pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
            if pth_files:
                models.append(name)
    return models


def run_training(
    model_name: str,
    audio_files: list,
    total_epoch: int,
    batch_size: int,
    progress=gr.Progress(),
) -> str:
    """Gradio callback for Tab 1 training."""
    if not model_name.strip():
        return "错误：请输入模型名称。"
    if not audio_files:
        return "错误：请上传至少一个声音样本文件。"

    dataset_dir = os.path.join(UPLOADS_DIR, model_name, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    for audio_file in audio_files:
        filename = os.path.basename(audio_file.name)
        shutil.copy(audio_file.name, os.path.join(dataset_dir, filename))

    log_lines = [
        f"模型名称: {model_name}",
        f"数据集目录: {dataset_dir}",
        f"训练轮数: {total_epoch}，批大小: {batch_size}",
        "开始训练...",
    ]

    progress(0.05, desc="开始训练...")
    try:
        model_path, index_path = train(
            model_name=model_name,
            dataset_path=dataset_dir,
            models_dir=MODELS_DIR,
            total_epoch=int(total_epoch),
            batch_size=int(batch_size),
        )
        progress(1.0, desc="训练完成！")
        log_lines += [
            "✅ 训练完成！",
            f"模型路径: {model_path}",
            f"索引路径: {index_path}",
        ]
    except Exception as e:
        log_lines.append(f"❌ 训练失败: {e}")

    return "\n".join(log_lines)


def run_inference(
    model_name: str,
    song_file,
    f0_up_key: int,
    vocals_volume: float,
    accompaniment_volume: float,
    progress=gr.Progress(),
):
    """Gradio callback for Tab 2 inference. Returns (original, converted, final, status)."""
    if not model_name:
        return None, None, None, "错误：请选择模型。"
    if song_file is None:
        return None, None, None, "错误：请上传歌曲文件。"

    model_dir = os.path.join(MODELS_DIR, model_name)
    pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    index_files = [f for f in os.listdir(model_dir) if f.endswith(".index")]

    if not pth_files:
        return None, None, None, f"错误：在 {model_dir} 中找不到 .pth 文件"
    if not index_files:
        return None, None, None, f"错误：在 {model_dir} 中找不到 .index 文件"

    model_path = os.path.join(model_dir, pth_files[0])
    index_path = os.path.join(model_dir, index_files[0])
    output_dir = os.path.join(OUTPUTS_DIR, model_name)

    progress(0.1, desc="分离人声中...")
    try:
        results = infer(
            song_path=song_file.name,
            model_path=model_path,
            index_path=index_path,
            output_dir=output_dir,
            f0_up_key=int(f0_up_key),
            vocals_volume=float(vocals_volume),
            accompaniment_volume=float(accompaniment_volume),
        )
        progress(1.0, desc="生成完成！")
        return (
            results["vocals_original"],
            results["vocals_converted"],
            results["final_output"],
            "✅ 生成完成！",
        )
    except Exception as e:
        return None, None, None, f"❌ 生成失败: {e}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="声音克隆翻唱系统") as demo:
        gr.Markdown("# 声音克隆 + 歌声生成系统")

        with gr.Tab("🎤 模型训练"):
            model_name_input = gr.Textbox(label="模型名称", placeholder="例如：jay_chou")
            audio_upload = gr.File(
                label="上传声音样本（支持多选，MP3/WAV，建议10–30分钟干净人声）",
                file_types=[".wav", ".mp3"],
                file_count="multiple",
            )
            with gr.Row():
                epoch_slider = gr.Slider(50, 500, value=200, step=50, label="训练轮数")
                batch_slider = gr.Slider(1, 16, value=4, step=1, label="批大小")
            train_btn = gr.Button("开始训练", variant="primary")
            train_log = gr.Textbox(label="训练日志", lines=12, interactive=False)
            train_btn.click(
                fn=run_training,
                inputs=[model_name_input, audio_upload, epoch_slider, batch_slider],
                outputs=[train_log],
            )

        with gr.Tab("🎵 翻唱生成"):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="选择模型",
                    choices=get_available_models(),
                    interactive=True,
                )
                refresh_btn = gr.Button("刷新模型列表")
            refresh_btn.click(
                fn=lambda: gr.update(choices=get_available_models()),
                outputs=[model_dropdown],
            )
            song_upload = gr.File(
                label="上传歌曲（MP3/WAV）",
                file_types=[".wav", ".mp3"],
            )
            f0_slider = gr.Slider(-12, 12, value=0, step=1, label="音调偏移（半音）；男翻女+5~+12，女翻男-5~-12")
            with gr.Row():
                vocal_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="人声音量")
                acc_vol = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="伴奏音量")
            infer_btn = gr.Button("开始生成", variant="primary")
            with gr.Row():
                audio_original = gr.Audio(label="原始人声")
                audio_converted = gr.Audio(label="转换后人声")
                audio_final = gr.Audio(label="最终混音")
            infer_status = gr.Textbox(label="状态", interactive=False)
            infer_btn.click(
                fn=run_inference,
                inputs=[model_dropdown, song_upload, f0_slider, vocal_vol, acc_vol],
                outputs=[audio_original, audio_converted, audio_final, infer_status],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
