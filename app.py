import streamlit as st
import math


st.set_page_config(page_title='pytorch calculator', layout='centered')

ERR_MSG_NUMBER = '숫자만 입력 가능하고, 값을 비워두면 안됩니다.'

st.write('''

# Conv2d, MaxPool2d Calculator
for PyTorch
[@teddylee777](https://github.com/teddylee777)
''')


@st.cache(allow_output_mutation=True)
def create_list():
    return []

layers = create_list()

container = st.container()

st.write('''
----
**Image input**
''')

col1, col2, col3 = st.columns(3)
img_width = col1.text_input(label="width", value="224")
img_height = col2.text_input(label="height", value="224")
img_channel = col3.text_input(label="channel", value="3")

out_img_width = img_width
out_img_height = img_height
out_img_channel = img_channel

def update_container():
    with container:
        st.write('''
        **Output**
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
        st.write(f'**output(channel, height, width): ({c}, {h}, {w})**')


def calculate_conv2d_output_size(image_size, kernel_size=3, stride=1, padding=0):
    return math.floor((image_size - kernel_size + 2 * padding) / stride) + 1

def calculate_maxpool2d_output_size(image_size, kernel_size=2, stride=2, padding=0):
    return math.floor((image_size + 2 * padding - (kernel_size -1) -1 ) / stride ) + 1

def calculate_output():
    width, height, channel = int(img_width), int(img_height), int(img_channel)
    for l in layers:
        if l['layer'] == 'Conv2d':
            out_channel = l['out_channel']
            kernel_size = l['kernel_size']
            stride = l['stride']
            padding = l['padding']
            width = calculate_conv2d_output_size(width, kernel_size, stride, padding)
            height = calculate_conv2d_output_size(height, kernel_size, stride, padding)
            channel = out_channel
        elif l['layer'] == 'MaxPool2d':
            kernel_size = l['kernel_size']
            stride = l['stride']
            padding = l['padding']
            width = calculate_maxpool2d_output_size(width, kernel_size, stride, padding)
            height = calculate_maxpool2d_output_size(height, kernel_size, stride, padding)
            channel = out_channel
    return width, height, channel


st.write('''
----
**Conv2d**
''')

conv_form = st.form(key='conv-form')

conv_col1, conv_col2, conv_col3, conv_col4 = conv_form.columns(4)
conv_input1 = conv_col1.text_input(label="out_channel", value="")
conv_input2 = conv_col2.text_input(label="kernel_size", value="3")
conv_input3 = conv_col3.text_input(label="stride(default=1)", value="1")
conv_input4 = conv_col4.text_input(label="padding(default=0)", value="0")

conv_submit = conv_form.form_submit_button('Add Conv2d Layer')

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


st.write('''
----
**MaxPool2d**
''')

mp2d_form = st.form(key='maxpool2d-form')

mp_col1, mp_col2, mp_col3 = mp2d_form.columns(3)
mp_input1 = mp_col1.text_input(label="kernel_size", value="2", key='mp_col1')
mp_input2 = mp_col2.text_input(label="stride", value="2", key='mp_col2')
mp_input3 = mp_col3.text_input(label="padding(default=0)", value="0", key='mp_col3')

mp2d_submit = mp2d_form.form_submit_button('Add MaxPool2d Layer')

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

if st.button('Clear'):
    layers.clear()







# conv2d = st.text_input(label="User Name", value="default value")
# radio_gender = st.radio(label="Gender", options=["Male", "Female"])
# check_1 = st.checkbox(label="agree", value=False)
# memo = st.text_area(label="memo", value="")

# if st.button("Confirm"):
#     con = st.container()
#     con.caption("Result")
#     con.write(f"User Name is {str(input_user_name)}")
#     con.write(str(radio_gender))
#     con.write(f"agree : {check_1}")
#     con.write(f"memo : {str(memo)}")