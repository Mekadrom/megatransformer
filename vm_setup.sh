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
# Use DEBIAN_FRONTEND=noninteractive to prevent interactive prompts
# -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" 
# will keep existing config files when possible
sudo apt update -y
echo "Running system upgrade. This might require your input for configuration changes..."
sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"

# Check if a reboot is needed
if [ -f /var/run/reboot-required ]; then
    echo "*** System update requires a reboot ***"
    read -p "Would you like to reboot now? (y/n): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        echo "The script will need to be run again after reboot."
        echo "Rebooting in 10 seconds..."
        sleep 10
        sudo reboot
        exit 0
    else
        echo "Continuing without reboot. Some changes may not take effect until system is rebooted."
    fi
fi

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
    
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        print_section "Installing Python requirements"
        pip3 install -r requirements.txt
    else
        echo "No requirements.txt found in the repository."
    fi
fi

git config --global credential.helper store
