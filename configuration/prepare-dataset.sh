sudo apt install -y python3-pip
pip3 install torch==1.9.0
pip3 install Pillow==8.2.0
pip3 install torchvision==0.10.0
python3 prepare_dataset.py
sudo docker pull xbfu1994/test:v4

