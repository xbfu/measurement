# Start a Kubernetes cluster without GPUs

## Launch instances
***Make sure that you always launch instances in N. Virginia (us-east-1).***

1. **Choose an AMI**: Ubuntu Server 18.04 LTS (HVM), SSD Volume Type
2. **Choose an Instance Type**: **t3.small**
3. **Configure Instance Details**: you may set the subnet to let your instances in the same subnet
4. **Add Storage**: 30 GiB
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
sudo apt-get install -y kubelet kubeadm kubectl docker.io
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
***Run `sudo kubectl logs <PodName>` on the master node to check the logs.***  
