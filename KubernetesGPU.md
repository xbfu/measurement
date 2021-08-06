# Start a Kubernetes cluster with GPUs

## Launch instances
***Make sure that you always launch instances in N. Virginia (us-east-1).***

1. **Choose an AMI**: Ubuntu Server 18.04 LTS (HVM), SSD Volume Type
2. **Choose an Instance Type**: **t3.xlarge** as the master node and **p2.xlarge** for worker nodes
3. **Configure Instance Details**: you may set the subnet to let your instances in the same subnet
4. **Add Storage**: 20 GiB for the master node and 40 GiB for each worker node
5. **Add Tags**: pass
6. **Configure Security Group**: use the existing security group (launch-wizard-1)
7. **Key pair**: choose an existing key pair if you have

## Install k8s and docker
#### On each node
```bash
sudo curl -O https://packages.cloud.google.com/apt/doc/apt-key.gpg
sudo apt-key add apt-key.gpg
echo " deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list
sudo apt-get update
sudo apt-get install -y kubelet=1.21.1-00 kubeadm=1.21.1-00 kubectl=1.21.1-00 docker.io
```

## Install NVIDIA driver on worker nodes
Ensure packages on the CUDA network repository have priority over the Canonical repository
``` bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-$distribution.pin
sudo mv cuda-$distribution.pin /etc/apt/preferences.d/cuda-repository-pin-600
```

Install the CUDA repository public GPG key
``` bash
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/7fa2af80.pub
```
Setup the CUDA network repository
``` bash
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
```
Update the APT repository cache and install the driver using the cuda-drivers meta-package
``` bash
sudo apt-get update
sudo apt-get -y install cuda-drivers
```
Check the GPU information
``` bash
nvidia-smi
```

## Modify hostname (optional)
*This step changes the hostname of an instance. It will show in Kubernetes nodes.*
```bash
sudo vim /etc/hostname
```
Replace the `<ip-172-31-xx-xx>` with what you want. Then reboot.

## Initialize a k8s cluster
#### On master node
```bash
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=<master_addr>
```
Replace the `<master_addr>` with the private IP of master node.

Config Kubernetes
```bash
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
sudo kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```
#### On  worker nodes
```bash
sudo kubeadm join <master_addr>:6443 --token 9nzp6b.gguo --discovery-token-ca-cert-hash sha256:1b9a48db383b
``` 
This command appears when you run `kubeadm init` on your master node. Copy it and run on each worker node.

***Run `sudo kubectl get node` on the master node to check if all the nodes are ready.***

## Initialize NVIDIA k8s-device plugin on worker nodes
Install NVIDIA k8s-device plugin on each worker node
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
Edit the docker daemon config file which is usually present at `/etc/docker/daemon.json` on each worker node
```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
Enabling GPU support in k8s on master node
``` bash
sudo kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.6.0/nvidia-device-plugin.yml
```
Then reboot worker nodes.

## Run sample code
#### On each node
Download the repo
```bash
cd ~ && git clone https://github.com/xbfu/measurement.git
```
#### On master node
Start a job
```bash
sudo kubectl apply -f measurement/yaml/run_MNIST.yaml
```
***Run `sudo kubectl get pod -o wide` on the master node to check if the job starts.***  
***Run `nvidia-smi` on the worker node where the job is running to check if it uses GPU.***  
***Run `sudo kubectl logs <PodName>` on the master node to check the logs. It should be completed within 20 seconds.***  
