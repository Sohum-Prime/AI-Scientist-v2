#!/bin/bash

# Copy the service file to the systemd directory
sudo cp /workspace/AI-Scientist-v2/ai-scientist-gradio.service /etc/systemd/system/

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable ai-scientist-gradio.service

# Start the service
sudo systemctl start ai-scientist-gradio.service

# Check the status of the service
sudo systemctl status ai-scientist-gradio.service