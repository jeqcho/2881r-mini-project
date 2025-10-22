#!/bin/bash
# RunPod Setup Script - Install Essential Development Tools
# This script installs: uv, apt utilities, git, GitHub CLI, Node.js/npm, and Claude Code
# NOTE: PUT YOUR HUGGINGFACE TOKEN AT SECRETS/huggingface.token BEFORE RUNNING THIS SCRIPT

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_status "Starting RunPod environment setup..."

# Update package lists
print_status "Updating package lists..."
apt update

# Install essential apt packages
print_status "Installing essential apt packages..."
apt install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    python3-pip \
    python3-venv \
    tmux \
    screen \
    htop \
    vim \
    nano \
    jq \
    unzip \
    zip

# Install uv (Python package manager)
print_status "Installing uv..."
if command_exists uv; then
    print_warning "uv is already installed, skipping..."
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Add to .bashrc for future sessions
    if ! grep -q ".cargo/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    fi
fi

# Install GitHub CLI
print_status "Installing GitHub CLI..."
if command_exists gh; then
    print_warning "GitHub CLI is already installed, skipping..."
else
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt update
    apt install -y gh
fi

# Install Node.js 20 and npm
print_status "Installing Node.js 20 and npm..."
if command_exists node; then
    NODE_VERSION=$(node --version)
    print_warning "Node.js ${NODE_VERSION} is already installed"
    
    # Check if version is less than 18
    if [[ "${NODE_VERSION}" < "v18" ]]; then
        print_status "Node.js version is too old, upgrading..."
        apt remove --purge -y nodejs npm
        rm -rf /usr/local/lib/node_modules
    else
        print_warning "Node.js version is sufficient, skipping upgrade..."
    fi
fi

if ! command_exists node || [[ "${NODE_VERSION}" < "v18" ]]; then
    # Clean any existing Node.js installations
    apt remove --purge -y nodejs npm libnode-dev || true
    apt autoremove -y
    
    # Install Node.js 20 from NodeSource
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs
fi

# Install Claude Code
print_status "Installing Claude Code..."
if command_exists claude; then
    print_warning "Claude Code is already installed, updating..."
    npm update -g @anthropic-ai/claude-code
else
    npm install -g @anthropic-ai/claude-code
fi

# Setup Git configuration
print_status "Setting up Git configuration..."
if [ -z "$(git config --global user.email)" ]; then
    print_warning "Git user email not set. Run: git config --global user.email 'your-email@example.com'"
fi
if [ -z "$(git config --global user.name)" ]; then
    print_warning "Git user name not set. Run: git config --global user.name 'Your Name'"
fi

# Create useful directories
print_status "Creating workspace directories..."
mkdir -p /workspace/projects
mkdir -p /workspace/models
mkdir -p /workspace/datasets
mkdir -p /workspace/.ssh
mkdir -p ~/.ssh

# Create symlinks for workspace directories
ln -sfn /workspace/projects ~/workspace
ln -sfn /workspace/models ~/models
ln -sfn /workspace/datasets ~/datasets

# Install additional Python tools
print_status "Installing useful Python packages..."
pip install --upgrade pip
pip install ipython jupyter notebook tensorboard

# Handle SSH keys with persistence in /workspace
print_status "Setting up SSH keys..."

# Check if SSH key exists in /workspace
if [ -f /workspace/.ssh/id_ed25519 ]; then
    print_status "Found existing SSH key in /workspace, copying to ~/.ssh..."
    cp /workspace/.ssh/id_ed25519 ~/.ssh/
    cp /workspace/.ssh/id_ed25519.pub ~/.ssh/
    chmod 600 ~/.ssh/id_ed25519
    chmod 644 ~/.ssh/id_ed25519.pub
    print_status "SSH public key:"
    cat ~/.ssh/id_ed25519.pub
else
    print_status "No SSH key found in /workspace, generating new one..."
    ssh-keygen -t ed25519 -C "runpod-key" -f ~/.ssh/id_ed25519 -N ""
    
    # Copy to /workspace for persistence
    cp ~/.ssh/id_ed25519 /workspace/.ssh/
    cp ~/.ssh/id_ed25519.pub /workspace/.ssh/
    chmod 600 /workspace/.ssh/id_ed25519
    chmod 644 /workspace/.ssh/id_ed25519.pub
    
    print_status "SSH key generated and saved to /workspace/.ssh/"
    print_status "SSH public key:"
    cat ~/.ssh/id_ed25519.pub
    print_warning "Add this key to your GitHub account: https://github.com/settings/keys"
fi


# Final verification
print_status "Verifying installations..."
echo ""
echo "Installation Summary:"
echo "===================="

# Check each tool
tools=("git" "gh" "node" "npm" "claude" "uv" "python3" "pip")
for tool in "${tools[@]}"; do
    if command_exists "$tool"; then
        version=$($tool --version 2>&1 | head -n1)
        echo -e "${GREEN}✓${NC} $tool: $version"
    else
        echo -e "${RED}✗${NC} $tool: Not installed"
    fi
done

echo ""
print_status "Setup complete! 🎉"
print_status "Run 'source ~/.bashrc' to load new aliases and functions"
pip install -U "huggingface_hub[cli]"

echo "huggingface-cli login" 
token=$(cat SECRETS/huggingface.token)
huggingface-cli login --token $token


git config --global user.name 'Jay Chooi' && git config --global user.email 'jeqin_chooi@college.harvard.edu'

# Print next steps
echo ""
echo "Test Claude Code: claude --version"
echo ""
echo "Note: SSH keys and configs are persisted in /workspace/"