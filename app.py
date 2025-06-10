import math
import pathlib
from bs4 import BeautifulSoup
import logging
import shutil
import os
import re
import json
from typing import Dict, List, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components

# 페이지 설정
st.set_page_config(
    page_title="PyTorch Layer Calculator - Advanced CNN Architecture Visualizer",
    layout="wide",
    page_icon="❤️",
    initial_sidebar_state="expanded",
)

# 커스텀 CSS
st.markdown(
    """
<style>
    /* 메인 컨테이너 스타일 */
    .main {
        padding: 2rem;
    }
    
    /* 카드 스타일 */
    .layer-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .layer-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* 레이어별 색상 */
    .conv2d-color { border-left: 5px solid #1f77b4; }
    .maxpool2d-color { border-left: 5px solid #ff7f0e; }
    .convtranspose2d-color { border-left: 5px solid #2ca02c; }
    .avgpool2d-color { border-left: 5px solid #d62728; }
    .batchnorm2d-color { border-left: 5px solid #9467bd; }
    
    /* 결과 박스 */
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        margin: 2rem 0;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background-color: #1e3a5f;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #152a47;
        transform: scale(1.05);
    }
    
    /* Add Layer 버튼 강조 스타일 */
    [data-testid="stForm"] .stButton > button {
        background-color: #ff6b6b;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(255, 107, 107, 0.3);
    }
    
    [data-testid="stForm"] .stButton > button:hover {
        background-color: #ff5252;
        box-shadow: 0 6px 8px rgba(255, 107, 107, 0.4);
    }
    
    /* 에러 메시지 스타일 */
    .stAlert {
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 상수 정의
ERR_MSG_NUMBER = "⚠️ 숫자만 입력 가능합니다. 모든 필드를 채워주세요."

# 레이어 색상 매핑
LAYER_COLORS = {
    "Conv2d": "#1f77b4",
    "MaxPool2d": "#ff7f0e",
    "ConvTranspose2d": "#2ca02c",
    "AvgPool2d": "#d62728",
    "BatchNorm2d": "#9467bd",
}

# 세션 상태 초기화
if "layers" not in st.session_state:
    st.session_state["layers"] = []

if "show_code" not in st.session_state:
    st.session_state["show_code"] = False

if "edit_mode" not in st.session_state:
    st.session_state["edit_mode"] = {}

# Google Analytics 코드 (기존 코드 유지)
if "ga" not in st.session_state:
    st.session_state["ga"] = True
    analytics_code = """<meta name="author" content="teddylee777">
    <meta name="description" content="Advanced PyTorch CNN layer calculator with visualization" />
    
    <!-- Twitter Card data -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="PyTorch Layer Calculator">
    <meta name="twitter:description" content="Advanced CNN architecture calculator with visualization">
    
    <!-- Open Graph data -->
    <meta property="og:title" content="PyTorch Layer Calculator" />
    <meta property="og:type" content="website" />
    <meta property="og:url" content="http://layer-calc.com" />
    <meta property="og:description" content="Advanced CNN architecture calculator with visualization" />
    
    <link rel="canonical" href="http://layer-calc.com">
    
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-X4423L75Z6"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-X4423L75Z6');
    </script>
    """

    # Analytics 코드 삽입 (기존 로직 유지)
    path_ind = os.path.dirname(st.__file__) + "/static/index.html"
    try:
        with open(path_ind, "r") as index_file:
            data = index_file.read()
            if len(re.findall("G-X4423L75Z6", data)) == 0:
                with open(path_ind, "w") as index_file_f:
                    newdata = re.sub("<head>", "<head>" + analytics_code, data)
                    index_file_f.write(newdata)
    except:
        pass


# 계산 함수들
def calculate_conv2d_output_size(
    image_size: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Conv2d 출력 크기 계산"""
    return math.floor(
        (image_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def calculate_convtranspose2d_output_size(
    image_size: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> int:
    """ConvTranspose2d 출력 크기 계산"""
    return (image_size - 1) * stride - 2 * padding + kernel_size + output_padding


def calculate_pool2d_output_size(
    image_size: int, kernel_size: int = 2, stride: int = 2, padding: int = 0
) -> int:
    """MaxPool2d/AvgPool2d 출력 크기 계산"""
    return math.floor((image_size + 2 * padding - kernel_size) / stride + 1)


def calculate_output(
    layers: List[Dict], input_dims: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """전체 네트워크의 출력 차원 계산"""
    batch, channel, height, width = input_dims

    for layer in layers:
        if layer["layer"] == "Conv2d":
            height = calculate_conv2d_output_size(
                height, layer["kernel_size"], layer["stride"], layer["padding"]
            )
            width = calculate_conv2d_output_size(
                width, layer["kernel_size"], layer["stride"], layer["padding"]
            )
            channel = layer["out_channel"]

        elif layer["layer"] in ["MaxPool2d", "AvgPool2d"]:
            height = calculate_pool2d_output_size(
                height, layer["kernel_size"], layer["stride"], layer["padding"]
            )
            width = calculate_pool2d_output_size(
                width, layer["kernel_size"], layer["stride"], layer["padding"]
            )

        elif layer["layer"] == "ConvTranspose2d":
            height = calculate_convtranspose2d_output_size(
                height, layer["kernel_size"], layer["stride"], layer["padding"]
            )
            width = calculate_convtranspose2d_output_size(
                width, layer["kernel_size"], layer["stride"], layer["padding"]
            )
            channel = layer["out_channel"]

        elif layer["layer"] == "BatchNorm2d":
            # BatchNorm2d doesn't change dimensions
            pass

    return batch, channel, height, width


def generate_pytorch_code(
    layers: List[Dict], input_dims: Tuple[int, int, int, int]
) -> str:
    """PyTorch 코드 생성"""
    code = f"""import torch
import torch.nn as nn

class GeneratedCNN(nn.Module):
    def __init__(self):
        super(GeneratedCNN, self).__init__()
        
        # Define layers
        self.layers = nn.Sequential("""

    in_channels = input_dims[1]

    for i, layer in enumerate(layers):
        if layer["layer"] == "Conv2d":
            code += f"""
            nn.Conv2d({in_channels}, {layer['out_channel']}, 
                     kernel_size={layer['kernel_size']}, 
                     stride={layer['stride']}, 
                     padding={layer['padding']}),
            nn.ReLU(inplace=True),"""
            in_channels = layer["out_channel"]

        elif layer["layer"] == "MaxPool2d":
            code += f"""
            nn.MaxPool2d(kernel_size={layer['kernel_size']}, 
                        stride={layer['stride']}, 
                        padding={layer['padding']}),"""

        elif layer["layer"] == "AvgPool2d":
            code += f"""
            nn.AvgPool2d(kernel_size={layer['kernel_size']}, 
                        stride={layer['stride']}, 
                        padding={layer['padding']}),"""

        elif layer["layer"] == "ConvTranspose2d":
            code += f"""
            nn.ConvTranspose2d({in_channels}, {layer['out_channel']}, 
                              kernel_size={layer['kernel_size']}, 
                              stride={layer['stride']}, 
                              padding={layer['padding']}),
            nn.ReLU(inplace=True),"""
            in_channels = layer["out_channel"]

        elif layer["layer"] == "BatchNorm2d":
            code += f"""
            nn.BatchNorm2d({in_channels}),"""

    code += """
        )
    
    def forward(self, x):
        return self.layers(x)

# Create model instance
model = GeneratedCNN()

# Test with input"""

    code += f"""
x = torch.randn({input_dims[0]}, {input_dims[1]}, {input_dims[2]}, {input_dims[3]})
output = model(x)
print(f"Output shape: {{output.shape}}")  # Expected: {calculate_output(layers, input_dims)}"""

    return code


def load_preset(preset_name: str):
    """프리셋 아키텍처 로드"""
    presets = {
        "LeNet-5": [
            {
                "layer": "Conv2d",
                "out_channel": 6,
                "kernel_size": 5,
                "stride": 1,
                "padding": 0,
            },
            {
                "layer": "AvgPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            {
                "layer": "Conv2d",
                "out_channel": 16,
                "kernel_size": 5,
                "stride": 1,
                "padding": 0,
            },
            {
                "layer": "AvgPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
        ],
        "AlexNet": [
            # Conv Block 1
            {
                "layer": "Conv2d",
                "out_channel": 96,
                "kernel_size": 11,
                "stride": 4,
                "padding": 2,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
            },
            # Conv Block 2
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 5,
                "stride": 1,
                "padding": 2,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
            },
            # Conv Block 3-5
            {
                "layer": "Conv2d",
                "out_channel": 384,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 384,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 3,
                "stride": 2,
                "padding": 0,
            },
        ],
        "VGG16": [
            # Block 1
            {
                "layer": "Conv2d",
                "out_channel": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            # Block 2
            {
                "layer": "Conv2d",
                "out_channel": 128,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 128,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            # Block 3
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            # Block 4
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            # Block 5
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
        ],
        "VGG19": [
            # Block 1
            {
                "layer": "Conv2d",
                "out_channel": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            # Block 2
            {
                "layer": "Conv2d",
                "out_channel": 128,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 128,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            # Block 3
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            # Block 4
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
            # Block 5
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "Conv2d",
                "out_channel": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
            },
        ],
        "ResNet Block": [
            {
                "layer": "Conv2d",
                "out_channel": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "BatchNorm2d",
                "out_channel": "-",
                "kernel_size": "-",
                "stride": "-",
                "padding": "-",
            },
            {
                "layer": "Conv2d",
                "out_channel": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "layer": "BatchNorm2d",
                "out_channel": "-",
                "kernel_size": "-",
                "stride": "-",
                "padding": "-",
            },
        ],
    }

    if preset_name in presets:
        st.session_state["layers"] = presets[preset_name]
        st.rerun()


def render_layer_card(layer: Dict, index: int):
    """레이어 카드 렌더링"""
    layer_type = layer["layer"]
    color_class = f"{layer_type.lower()}-color"

    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(
                f"""
            <div class="layer-card {color_class}">
                <h4 style="margin: 0; font-size: 1rem;">{layer_type} - Layer {index + 1}</h4>
                <p style="margin: 0.25rem 0; font-size: 0.9rem;">
                Out: {layer.get('out_channel', '-')} | 
                K: {layer.get('kernel_size', '-')} | 
                S: {layer.get('stride', '-')} | 
                P: {layer.get('padding', '-')}
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            if st.button(f"✏️", key=f"edit_{index}"):
                st.session_state["edit_mode"][index] = not st.session_state[
                    "edit_mode"
                ].get(index, False)

        with col3:
            if st.button(f"🗑️", key=f"delete_{index}"):
                st.session_state["layers"].pop(index)
                st.rerun()

        # 편집 모드
        if st.session_state["edit_mode"].get(index, False):
            edit_layer(layer, index)


def edit_layer(layer: Dict, index: int):
    """레이어 편집 폼"""
    with st.form(key=f"edit_form_{index}"):
        cols = st.columns(4)

        if layer["layer"] in ["Conv2d", "ConvTranspose2d"]:
            new_out_channel = cols[0].number_input(
                "Out Channels", value=layer["out_channel"], min_value=1
            )
            new_kernel_size = cols[1].number_input(
                "Kernel Size", value=layer["kernel_size"], min_value=1
            )
            new_stride = cols[2].number_input(
                "Stride", value=layer["stride"], min_value=1
            )
            new_padding = cols[3].number_input(
                "Padding", value=layer["padding"], min_value=0
            )
        else:
            new_kernel_size = cols[1].number_input(
                "Kernel Size", value=layer["kernel_size"], min_value=1
            )
            new_stride = cols[2].number_input(
                "Stride", value=layer["stride"], min_value=1
            )
            new_padding = cols[3].number_input(
                "Padding", value=layer["padding"], min_value=0
            )

        if st.form_submit_button("Save Changes"):
            if layer["layer"] in ["Conv2d", "ConvTranspose2d"]:
                layer["out_channel"] = int(new_out_channel)
            layer["kernel_size"] = int(new_kernel_size)
            layer["stride"] = int(new_stride)
            layer["padding"] = int(new_padding)
            st.session_state["edit_mode"][index] = False
            st.rerun()


# 헤더
st.markdown(
    """
# ❤️ PyTorch Layer Calculator

made by [teddynote](https://youtube.com/c/teddynote)

### Advanced CNN Architecture Designer

Design your CNN architecture with real-time dimension calculations.

[![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-black)](https://github.com/teddylee777/pytorch-layer-calculator)
"""
)

# 사이드바
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")

    # 액션 버튼들
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["layers"] = []
            st.rerun()

    with col2:
        if st.button("↩️ Undo", use_container_width=True):
            if len(st.session_state["layers"]) > 0:
                st.session_state["layers"].pop()
                st.rerun()

    st.markdown("---")

    # 설정 옵션
    st.markdown("## ⚙️ Settings")
    show_formula = st.checkbox("Show formulas", value=True)

    # 코드 생성 옵션
    if st.button("🔧 Generate PyTorch Code", use_container_width=True):
        st.session_state["show_code"] = not st.session_state["show_code"]

    st.markdown("---")

    # 프리셋
    st.markdown("## 📋 Presets")
    preset = st.selectbox(
        "Load preset architecture",
        ["None", "LeNet-5", "AlexNet", "VGG16", "VGG19", "ResNet Block"],
    )

    if st.button("Load Preset", use_container_width=True):
        load_preset(preset)

# 메인 컨텐츠
main_container = st.container()

# 입력 섹션
with main_container:
    st.markdown("### 📥 Input Dimensions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, value=1, step=1)
    with col2:
        img_channel = st.number_input("Channels", min_value=1, value=3, step=1)
    with col3:
        img_height = st.number_input("Height", min_value=1, value=224, step=1)
    with col4:
        img_width = st.number_input("Width", min_value=1, value=224, step=1)

# 레이어 추가 섹션
st.markdown("### 🏗️ Add Layers")

tabs = st.tabs(["Conv2d", "MaxPool2d", "ConvTranspose2d", "AvgPool2d", "BatchNorm2d"])

# Conv2d 탭
with tabs[0]:
    with st.form(key="conv2d-form"):
        cols = st.columns(4)
        conv_out_channel = cols[0].number_input(
            "Out Channels", min_value=1, value=32, step=1
        )
        conv_kernel_size = cols[1].number_input(
            "Kernel Size", min_value=1, max_value=11, value=3, step=1
        )
        conv_stride = cols[2].number_input(
            "Stride", min_value=1, max_value=5, value=1, step=1
        )
        conv_padding = cols[3].number_input(
            "Padding", min_value=0, max_value=5, value=0, step=1
        )

        if show_formula:
            st.latex(
                r"Output = \lfloor \frac{Input + 2 \times Padding - Kernel}{Stride} \rfloor + 1"
            )

        if st.form_submit_button("➕ Add Conv2d Layer", use_container_width=True):
            layer = {
                "layer": "Conv2d",
                "out_channel": int(conv_out_channel),
                "kernel_size": int(conv_kernel_size),
                "stride": int(conv_stride),
                "padding": int(conv_padding),
            }
            st.session_state["layers"].append(layer)
            st.rerun()

# MaxPool2d 탭
with tabs[1]:
    with st.form(key="maxpool2d-form"):
        cols = st.columns(3)
        mp_kernel_size = cols[0].number_input(
            "Kernel Size", min_value=1, max_value=5, value=2, step=1
        )
        mp_stride = cols[1].number_input(
            "Stride", min_value=1, max_value=5, value=2, step=1
        )
        mp_padding = cols[2].number_input(
            "Padding", min_value=0, max_value=2, value=0, step=1
        )

        if show_formula:
            st.latex(
                r"Output = \lfloor \frac{Input + 2 \times Padding - Kernel}{Stride} \rfloor + 1"
            )

        if st.form_submit_button("➕ Add MaxPool2d Layer", use_container_width=True):
            layer = {
                "layer": "MaxPool2d",
                "out_channel": "-",
                "kernel_size": int(mp_kernel_size),
                "stride": int(mp_stride),
                "padding": int(mp_padding),
            }
            st.session_state["layers"].append(layer)
            st.rerun()

# ConvTranspose2d 탭
with tabs[2]:
    with st.form(key="convtranspose2d-form"):
        cols = st.columns(4)
        ct_out_channel = cols[0].number_input(
            "Out Channels", min_value=1, value=32, step=1
        )
        ct_kernel_size = cols[1].number_input(
            "Kernel Size", min_value=1, max_value=11, value=3, step=1
        )
        ct_stride = cols[2].number_input(
            "Stride", min_value=1, max_value=5, value=1, step=1
        )
        ct_padding = cols[3].number_input(
            "Padding", min_value=0, max_value=5, value=0, step=1
        )

        if show_formula:
            st.latex(r"Output = (Input - 1) \times Stride - 2 \times Padding + Kernel")

        if st.form_submit_button(
            "➕ Add ConvTranspose2d Layer", use_container_width=True
        ):
            layer = {
                "layer": "ConvTranspose2d",
                "out_channel": int(ct_out_channel),
                "kernel_size": int(ct_kernel_size),
                "stride": int(ct_stride),
                "padding": int(ct_padding),
            }
            st.session_state["layers"].append(layer)
            st.rerun()

# AvgPool2d 탭
with tabs[3]:
    with st.form(key="avgpool2d-form"):
        cols = st.columns(3)
        ap_kernel_size = cols[0].number_input(
            "Kernel Size", min_value=1, max_value=5, value=2, step=1
        )
        ap_stride = cols[1].number_input(
            "Stride", min_value=1, max_value=5, value=2, step=1
        )
        ap_padding = cols[2].number_input(
            "Padding", min_value=0, max_value=2, value=0, step=1
        )

        if st.form_submit_button("➕ Add AvgPool2d Layer", use_container_width=True):
            layer = {
                "layer": "AvgPool2d",
                "out_channel": "-",
                "kernel_size": int(ap_kernel_size),
                "stride": int(ap_stride),
                "padding": int(ap_padding),
            }
            st.session_state["layers"].append(layer)
            st.rerun()

# BatchNorm2d 탭
with tabs[4]:
    with st.form(key="batchnorm2d-form"):
        st.info("BatchNorm2d normalizes the input and doesn't change dimensions.")

        if st.form_submit_button("➕ Add BatchNorm2d Layer", use_container_width=True):
            layer = {
                "layer": "BatchNorm2d",
                "out_channel": "-",
                "kernel_size": "-",
                "stride": "-",
                "padding": "-",
            }
            st.session_state["layers"].append(layer)
            st.rerun()

# 레이어 스택 표시
if st.session_state["layers"]:
    st.markdown("### 📊 Layer Stack")

    # 레이어 카드 표시
    for i, layer in enumerate(st.session_state["layers"]):
        render_layer_card(layer, i)

    # 출력 차원 계산
    input_dims = (int(batch_size), int(img_channel), int(img_height), int(img_width))
    output_dims = calculate_output(st.session_state["layers"], input_dims)

    # 결과 표시
    st.markdown(
        f"""
    <div class="result-box">
        <h2>🎯 Output Dimensions</h2>
        <h3>Batch × Channels × Height × Width</h3>
        <h1>{output_dims[0]} × {output_dims[1]} × {output_dims[2]} × {output_dims[3]}</h1>
        <p>Total parameters: ~{sum([l.get('out_channel', 0) * l.get('kernel_size', 0)**2 for l in st.session_state['layers'] if isinstance(l.get('out_channel'), int)]):,}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # PyTorch 코드 생성
    if st.session_state["show_code"]:
        st.markdown("### 💻 Generated PyTorch Code")
        code = generate_pytorch_code(st.session_state["layers"], input_dims)
        st.code(code, language="python")

        # 코드 복사 버튼
        st.download_button(
            label="📥 Download Code",
            data=code,
            file_name="generated_cnn.py",
            mime="text/x-python",
        )

    # JSON 내보내기/가져오기
    st.markdown("### 💾 Save/Load Configuration")
    col1, col2 = st.columns(2)

    with col1:
        config_json = json.dumps(st.session_state["layers"], indent=2)
        st.download_button(
            label="📥 Export Configuration",
            data=config_json,
            file_name="layer_config.json",
            mime="application/json",
        )

    with col2:
        uploaded_file = st.file_uploader("📤 Import Configuration", type=["json"])
        if uploaded_file is not None:
            try:
                loaded_config = json.load(uploaded_file)
                st.session_state["layers"] = loaded_config
                st.success("Configuration loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading configuration: {str(e)}")

else:
    st.info(
        "👆 Add layers using the forms above to start building your CNN architecture!"
    )

# 푸터
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #888;'>
    <a href='https://pytorch.org/docs/stable/nn.html'>PyTorch Documentation</a> | 
    <a href='https://github.com/teddylee777/pytorch-layer-calculator'>GitHub</a>
</div>
""",
    unsafe_allow_html=True,
)

# Streamlit 기본 요소 숨기기
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
