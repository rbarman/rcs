'''
AgentClient.py

AgentClient is a socket that connects and sends messages to an AgentServer 
Agent's represent posting behavior of users on social media sites.
	- Messages are generated on https://github.com/minimaxir/textgenrnn models
	- Can instantiate a user with pretrained weights that work with textgenrnn architecture

Example usage:

	# run agent with default textgenrnn model
	python AgentClient.py -p 9999 
	# run agent with some pretrained weights (must pass path to weights!)
	python AgentClient.py -p 9999 -w ~/Desktop/my_weights.hdf5

'''

import socket
import argparse
import json
import random
from textgenrnn import textgenrnn

class AgentClient:

	def __init__(self,port,weights):

		self.port = port
		self.socket = socket.socket()
		# connect Agent to localhost + specified port
		self.socket.connect(('localhost',self.port))
		self.weights = weights
		self.textgen = textgenrnn(self.weights)

		print('----------------------------------')
		print(f'Using weights: {self.weights}')
		print(f'connecting to {self.port}')
		print('----------------------------------')

	def run(self):
		''' Client is connected with server and sending messages
				Generating messages with textenrnn
		'''

		# Client gets an introductory message from server after connecting
		s_msg = self.socket.recv(1024).decode()
		print(f'{s_msg}')

		while True:
			# send a generated message TO server
			generated_text = self.textgen.generate(return_as_list = True, temperature = 1.0)[0].encode('utf-8')
			print(generated_text)
			self.socket.send(generated_text)

if __name__ == '__main__':

	# set up arg parser
	parser = argparse.ArgumentParser(description='Agent Client')
	parser.add_argument('-p','--port',help='Port for agent to send messages to ', type=int, default=9572)
	parser.add_argument('-w','--weights',help='hd5 weights for textgenrnn models')
	args = parser.parse_args()

	# Start client to send messages on localhost + specified port and GSP
	client = AgentClient(args.port, args.weights)
	client.run()