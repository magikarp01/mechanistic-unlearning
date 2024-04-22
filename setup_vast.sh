python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
rm -r tasks/
git clone -b aaquib-pythia-sports https://github.com/magikarp01/tasks.git --single-branch

echo "Setting up VSCode..."
# Defining the vscode variable with the path to the VSCode executable
vscode_path=$(ls -td ~/.vscode-server/bin/*/bin/remote-cli/code | head -1)
vscode="$vscode_path"

# Append vscode path to .bashrc for future use
echo 'alias code="'$vscode'"' >> ~/.bashrc
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc

# Update the system and install jq
sudo apt-get update
sudo apt-get install -y jq

# Install recommended VSCode extensions
jq -r '.recommendations[]' ~/.vscode-server/extensions/extensions.json | while read extension; do "$vscode" --install-extension "$extension"; done
