# PyTorch Layer Calculator 🧮

An advanced web tool for designing and visualizing CNN architectures with real-time dimension calculations.

**Live Demo:** http://layer-calc.com

## ✨ Features

### Core Functionality
- **Layer Support**: Conv2d, MaxPool2d, ConvTranspose2d, AvgPool2d, BatchNorm2d
- **Real-time Calculations**: Instant dimension updates as you build
- **Batch Dimension**: Full support for batch size in calculations

### Advanced Features
- **💻 Code Generation**: Auto-generate PyTorch model code
- **📋 Presets**: Load common architectures (LeNet-5, AlexNet, VGG, ResNet)
- **💾 Import/Export**: Save and load configurations as JSON
- **✏️ Layer Editing**: Modify parameters after adding layers
- **📊 Parameter Count**: Automatic calculation of model parameters

### UI/UX Enhancements
- **Modern Design**: Compact card-based layout with hover effects
- **Smart Inputs**: Number inputs for precise parameter control
- **Formula Display**: LaTeX formulas for each layer type
- **Color Coding**: Visual differentiation between layer types
- **Responsive Layout**: Wide layout for better visibility

![demo](./demo.gif)

## Local Development

### Using UV (Recommended)

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project
uv pip install -e .

# Run the app
streamlit run app.py
```

### Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at http://localhost:8501

## Deployment

This app is deployed on Heroku. To deploy your own instance:

```bash
git push heroku master
```

---
Built with [Streamlit](https://streamlit.io)
