import os
from redis import Redis
"""
=============================================================================
PROJECT SETUP & DEPENDENCIES INSTRUCTIONS
=============================================================================
# Run in Windows PowerShell (as Administrator), then restart your computer:
wsl --install
1. START REDIS SERVER (Choose one method based on your setup):
   
# Run in your WSL (Ubuntu) terminal after reboot:
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb jammy main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install -y redis-stack-server
sudo service redis-stack-server start
redis-cli MODULE LIST
=============================================================================
"""
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
# Single connection point for both files
redis_client = Redis.from_url(REDIS_URL, decode_responses=True)