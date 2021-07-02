sudo curl -O https://packages.cloud.google.com/apt/doc/apt-key.gpg
sudo apt-key add apt-key.gpg
echo " deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl docker.io

