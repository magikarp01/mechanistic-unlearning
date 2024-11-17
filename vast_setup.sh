#
extensions=(
    mikoz.black-py         
    GitHub.copilot       
    GitHub.copilot-chat
    ms-toolsai.jupyter
    ms-toolsai.vscode-jupyter-cell-tags
    ms-toolsai.vscode-jupyter-slideshow
    ms-python.vscode-pylance
    ms-python.python
    ms-python.debugpy
)

# Install each extension
for extension in "${extensions[@]}"; do
    echo "Installing $extension..."
    code --install-extension "$extension" --force
done

echo "All extensions installed successfully!"

cd ~/
python -m venv venv
source venv/bin/activate
cd ~/mechanistic-unlearning/
pip install -r requirements.txt
cd tasks/
git clone https://github.com/magikarp01/tasks.git

echo "Setup Complete"
