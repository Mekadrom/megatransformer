#!/bin/bash
set -e

# Function to print section headers
print_section() {
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
}

# Update and upgrade system packages
print_section "Updating system packages"
sudo apt update -y && sudo apt upgrade -y

# Install NVIDIA CUDA toolkit
print_section "Installing NVIDIA CUDA toolkit"
sudo apt install -y nvidia-cuda-toolkit

# Generate SSH key
print_section "Generating SSH key"
ssh-keygen -t ed25519 -f ~/.ssh/github_key -N ""

# Display the public key
print_section "SSH Public Key"
echo "Add this public key to your GitHub account:"
cat ~/.ssh/github_key.pub

# Add the key to SSH config
cat >> ~/.ssh/config << EOF
Host github.com
    IdentityFile ~/.ssh/github_key
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config

# Wait for user confirmation
echo ""
read -p "Press Enter after you've added the SSH key to your GitHub account... "

# Get the Git repository URL
print_section "Repository setup"
read -p "Enter the Git repository URL (ssh format, e.g., git@github.com:username/repo.git): " REPO_URL

# Clone the git repository
print_section "Cloning repository"
if [ -z "$REPO_URL" ]; then
    echo "No repository URL provided, skipping clone."
else
    git clone "$REPO_URL" ~/project
    cd ~/project
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        print_section "Installing Python requirements"
        pip3 install -r requirements.txt
    else
        echo "No requirements.txt found in the repository."
    fi
fi

git config --global credential.helper store

# Hugging Face login
print_section "Hugging Face login"
echo "You will need your Hugging Face token for the next step."
echo "Get your token from: https://huggingface.co/settings/tokens"
read -p "Press Enter when you're ready to proceed with Hugging Face login... "
huggingface-cli login

print_section "Setup complete!"
echo "Your environment has been configured successfully."
