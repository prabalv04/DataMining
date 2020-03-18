import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

train_loader = torch.utils.data.DataLoader(
					torchvision.datasets.MNIST("/Users/dineshvashisht/DataMining/Assignments",
					download=True, train=True, transform=torchvision.transforms.ToTensor()),
					batch_size=1, shuffle=True)

test_loader = torch.utils.data.DataLoader(
					torchvision.datasets.MNIST("/Users/dineshvashisht/DataMining/Assignments",
					download=True, train=False, transform=torchvision.transforms.ToTensor()),
					batch_size=1, shuffle=True)

def calculate_loss(w, w0, x, y):
	vec = torch.matmul(w,x)+w0-y

	return torch.norm(vec).item()

def gradient(w, w0, x, y):
	w1 = torch.matmul(torch.matmul(w, x), x.transpose(0,1))
	w2 = 2*torch.matmul(w0, x.transpose(0, 1))
	w3 = -2*torch.matmul(y, x.transpose(0,1))

	w01 = 2*torch.matmul(w, x)
	w02 = 2*w0
	w03 = -2*y

	return [w1+w2+w3, w01+w02+w03]

def create_x(x):
	return x.view(28*28, 1)

def create_y(y):
	ret_y = torch.zeros((10, 1))
	ret_y[y] = 1

	return ret_y


epochs = 1000
step_size = 0.0001
n_train = len(train_loader)
n_test = len(test_loader)

w = torch.rand((10, 28*28))
w0 = torch.rand((10, 1))

training_loss_arr = []
test_loss_arr = []

for epoch in range(epochs):
	print(epoch)
	train_loss = 0
	test_loss = 0
	tot_gradient_w = torch.zeros((10, 28*28))
	tot_gradient_w0 = torch.zeros((10, 1))

	for i, data in enumerate(train_loader):
		x, y = data
		x = create_x(x)
		y = create_y(y)
		gradient_w, gradient_w0 = gradient(w, w0, x, y)
		tot_gradient_w += gradient_w
		tot_gradient_w0 += gradient_w0

		train_loss += calculate_loss(w, w0, x, y)

	for i, data in enumerate(test_loader):
		x, y = data
		x = create_x(x)
		y = create_y(y)
		test_loss += calculate_loss(w, w0, x, y)

	w = w-step_size*(tot_gradient_w/n_train)
	w0 = w0-step_size*(tot_gradient_w0/n_train)


	training_loss_arr.append(train_loss/n_train)
	test_loss_arr.append(test_loss/n_test)

	print(training_loss_arr[-1])
	

