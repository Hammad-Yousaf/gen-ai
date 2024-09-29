import streamlit as st
import torch
import sys
from PIL import Image
import os
import numpy as np

sys.path.append('C:/Users/zeesh/Documents/GEN AI/Image_Manipulation/streamlit_app/stylegan2_ada_pytorch')

from stylegan2_ada_pytorch.dnnlib import util
from stylegan2_ada_pytorch import legacy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    with util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G

G = load_model()

# Ensure the folders exist
os.makedirs('generated_images', exist_ok=True)
os.makedirs('manipulated_images', exist_ok=True)
os.makedirs('interpolated_images', exist_ok=True)

if 'latent_vector' not in st.session_state:
    st.session_state.latent_vector = None
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'manipulated_image' not in st.session_state:
    st.session_state.manipulated_image = None
if 'interpolated_image' not in st.session_state:
    st.session_state.interpolated_image = None

def generate_latent_vector():
    return torch.randn(1, G.z_dim).to(device)

if st.button('Generate Image'):
    z = generate_latent_vector()  
    st.session_state.latent_vector = z  

    img = G(z, None)
    img_np = (img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5
    img_np = img_np.astype(np.uint8)[0]
    img_pil = Image.fromarray(img_np)

    # Save the generated image to the folder
    img_pil.save(f'generated_images/generated_image_{np.random.randint(1000)}.png')
    
    st.session_state.generated_image = img_pil  
    st.session_state.manipulated_image = None  
    st.session_state.interpolated_image = None 

style_strength = st.slider('Style Strength', -3.0, 3.0, 0.0)
style_dimension = st.slider('Style Dimension', 0, G.mapping.num_layers - 1, 0)

if st.session_state.latent_vector is not None:
    z = st.session_state.latent_vector  
    ws = G.mapping(z, None)
    ws[:, style_dimension] += style_strength

    if st.button('Generate Manipulated Image'):
        img = G.synthesis(ws)
        img_np = (img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5
        img_np = img_np.astype(np.uint8)[0]
        img_pil = Image.fromarray(img_np)

        # Save the manipulated image to the folder
        img_pil.save(f'manipulated_images/manipulated_image_{np.random.randint(1000)}.png')
        
        st.session_state.manipulated_image = img_pil  

if st.session_state.latent_vector is not None:
    z1 = st.session_state.latent_vector  
    z2 = generate_latent_vector()  

    alpha = st.slider('Interpolation Factor', 0.0, 1.0, 0.5)  
    z_interp = alpha * z1 + (1 - alpha) * z2  
    if st.button('Generate Interpolated Image'):
        img = G(z_interp, None)
        img_np = (img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5
        img_np = img_np.astype(np.uint8)[0]
        img_pil = Image.fromarray(img_np)

        # Save the interpolated image to the folder
        img_pil.save(f'interpolated_images/interpolated_image_{np.random.randint(1000)}.png')
        
        st.session_state.interpolated_image = img_pil  

if st.session_state.generated_image:
    st.image(st.session_state.generated_image, caption='Generated Image', use_column_width=True)

if st.session_state.manipulated_image:
    st.image(st.session_state.manipulated_image, caption='Manipulated Image', use_column_width=True)

if st.session_state.interpolated_image:
    st.image(st.session_state.interpolated_image, caption='Interpolated Image', use_column_width=True)

if st.button('Clear'):
    st.session_state.generated_image = None
    st.session_state.manipulated_image = None
    st.session_state.interpolated_image = None 
    st.session_state.latent_vector = None   
