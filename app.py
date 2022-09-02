import math
import pathlib
from bs4 import BeautifulSoup
import logging
import shutil
import os
import re

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title='pytorch calculator', layout='centered')

ERR_MSG_NUMBER = 'Only NUMBERs are allowed as an input and it cannot be omitted.'

st.write('''

# Conv2d, MaxPool2d Calculator
for PyTorch
[@teddylee777](https://github.com/teddylee777)
''')


if 'layers' not in st.session_state:
    st.session_state['layers'] = []

if 'action' not in st.session_state:
    st.session_state['action'] = []

layers = st.session_state['layers']

image_label = st.empty()
image_section = st.empty()

main_tab = st.empty()

container = st.container()

def btn2_onclick():
    if len(layers) > 0:
        layers.pop(len(layers)-1)
        update_container()


with st.sidebar:
    # btn1, btn2 = st.columns(2)
    st.write('## 💡 Clear Everything')
    st.button('Clear', on_click=lambda: layers.clear(), )
    st.write('## ❌ Undo Action')
    st.button('Undo', on_click=btn2_onclick)


image_label.write('''
----
**Image input**
''')

col1, col2, col3 = image_section.columns(3)
img_width = col1.text_input(label="width", value="224")
img_height = col2.text_input(label="height", value="224")
img_channel = col3.text_input(label="channel", value="3")

out_img_width = img_width
out_img_height = img_height
out_img_channel = img_channel

def update_container():
    with container:
        st.write('''
        ----
        **📌 Output**
        ''')
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.write('**layer**')
        h2.write('**out_channel**')
        h3.write('**kernel_size**')
        h4.write('**stride**')
        h5.write('**padding**')

        for l in layers:
            cols = st.columns(5)
            cols[0].write(f"{l['layer']}")
            cols[1].write(f"{l['out_channel']}")
            cols[2].write(f"{l['kernel_size']}")
            cols[3].write(f"{l['stride']}")
            cols[4].write(f"{l['padding']}")

        w, h, c = calculate_output()
        st.write(f'====================================================')
        st.write(f'**🔆 output(channel, height, width): ({c}, {h}, {w})**')


def calculate_conv2d_output_size(image_size, kernel_size=3, stride=1, padding=0):
    return math.floor((image_size - kernel_size + 2 * padding) / stride) + 1

def calculate_maxpool2d_output_size(image_size, kernel_size=2, stride=2, padding=0):
    return math.floor((image_size + 2 * padding - (kernel_size -1) -1 ) / stride ) + 1

def calculate_output():
    width, height, channel = int(img_width), int(img_height), int(img_channel)
    global_out_channel = channel
    for l in layers:
        if l['layer'] == 'Conv2d':
            out_channel = l['out_channel']
            kernel_size = l['kernel_size']
            stride = l['stride']
            padding = l['padding']
            width = calculate_conv2d_output_size(width, kernel_size, stride, padding)
            height = calculate_conv2d_output_size(height, kernel_size, stride, padding)
            channel = out_channel
            global_out_channel = channel
        elif l['layer'] == 'MaxPool2d':
            kernel_size = l['kernel_size']
            stride = l['stride']
            padding = l['padding']
            width = calculate_maxpool2d_output_size(width, kernel_size, stride, padding)
            height = calculate_maxpool2d_output_size(height, kernel_size, stride, padding)
            channel = global_out_channel
    return width, height, channel


tab1, tab2 = main_tab.tabs(["Conv2d", "MaxPool2d"])

tab1.subheader("Conv2d")
tab1.write('''
```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```
[Go to Document](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
''')
tab2.subheader("MaxPool2d")
tab2.write('''
```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0)
```
[Go to Document](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool2d#torch.nn.MaxPool2d)
''')

conv_form = tab1.form(key='conv-form')

conv_col1, conv_col2, conv_col3, conv_col4 = conv_form.columns(4)
conv_input1 = conv_col1.text_input(label="out_channel", value="")
conv_input2 = conv_col2.text_input(label="kernel_size", value="3")
conv_input3 = conv_col3.text_input(label="stride(default=1)", value="1")
conv_input4 = conv_col4.text_input(label="padding(default=0)", value="0")

conv_submit = conv_form.form_submit_button('✔️ Add Layer')

if conv_submit:
    conv2d = dict()
    conv2d['layer'] = 'Conv2d'
    try:
        conv2d['out_channel'] = int(conv_input1)
        conv2d['kernel_size'] = int(conv_input2)
        conv2d['stride'] = int(conv_input3)
        conv2d['padding'] = int(conv_input4)
        layers.append(conv2d)
        update_container()
    except ValueError:
        st.error(ERR_MSG_NUMBER)
        update_container()


mp2d_form = tab2.form(key='maxpool2d-form')

mp_col1, mp_col2, mp_col3 = mp2d_form.columns(3)
mp_input1 = mp_col1.text_input(label="kernel_size", value="2", key='mp_col1')
mp_input2 = mp_col2.text_input(label="stride", value="2", key='mp_col2')
mp_input3 = mp_col3.text_input(label="padding(default=0)", value="0", key='mp_col3')

mp2d_submit = mp2d_form.form_submit_button('✔️ Add Layer')

if mp2d_submit:
    try:
        mp2d = dict()
        mp2d['layer'] = 'MaxPool2d'
        mp2d['out_channel'] = '\-'
        mp2d['kernel_size'] = int(mp_input1)
        mp2d['stride'] = int(mp_input2)
        mp2d['padding'] = int(mp_input3)
        layers.append(mp2d)
        update_container()
    except ValueError:
        st.error(ERR_MSG_NUMBER)
        update_container()

st.write('''
----
''')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


def inject_ga():
    GA_ID = "google_analytics"

    # Note: Please replace the id from G-XXXXXXXXXX to whatever your
    # web application's id is. You will find this in your Google Analytics account
    
    GA_JS = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-X4423L75Z6"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());

gtag('config', 'G-X4423L75Z6');
</script>
    """

    # Insert the script in the head tag of the static template inside your virtual
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    logging.info(f'editing {index_path}')
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    if not soup.find(id=GA_ID):  # if cannot find tag
        bck_index = index_path.with_suffix('.bck')
        if bck_index.exists():
            shutil.copy(bck_index, index_path)  # recover from backup
        else:
            shutil.copy(index_path, bck_index)  # keep a backup
        html = str(soup)
        new_html = html.replace('<head>', '<head>\n' + GA_JS)
        index_path.write_text(new_html)

inject_ga()

components.html('''
<head>
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-X4423L75Z6"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-X4423L75Z6');
</script>
</head>
''', width=0, height=0)

anlytcs_code = """<script async src="https://www.googletagmanager.com/gtag/js?id=G-X4423L75Z6"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-X4423L75Z6');
</script>"""

# Fetch the path of the index.html file
path_ind = os.path.dirname(st.__file__)+'/static/index.html'

# Open the file
with open(path_ind, 'r') as index_file:
    data=index_file.read()

    # Check whether there is GA script
    if len(re.findall('UA-', data))==0:

        # Insert Script for Google Analytics
        with open(path_ind, 'w') as index_file_f:

            # The Google Analytics script should be pasted in the header of the HTML file
            newdata=re.sub('<head>','<head>'+anlytcs_code,data)

            index_file_f.write(newdata)