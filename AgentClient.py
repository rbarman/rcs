'''
AgentClient.py
	Rohan Barman 4/12/2020

AgentClient is a socket that connects and sends/received messages to an AgentServer 
It can either be run in simulation mode where an Agent sends generated messages to the server
or in non simulation mode where an end user can send free text via the terminal to the server. 

AgentClients can also be initialized with a json containing GSP information. IT should be in atleast in format of:
	{
		,'Goals' :<>
		,'Standards' :<>
		,'Preferences' :<>
	}

Example usage:

	Default:
		port (p) = 9572
		simulation (s) = False

	# run agent with in non simulation mode on port 9999
	python AgentClient.py -p 9999

	# run agent with in non simulation mode on port 9999 and specified path to file with GSP
	python AgentClient.py -p 9999 -GSP GSP.json

	# run agent with in simulation mode on port 9999 and specified path to file with GSP
	python AgentClient.py -p 9999 -GSP GSP.json  -s True > simulated_log.txt	

	# run agent with in simulation mode on port 9999
	python AgentClient.py -p 9999 -GSP GSP.json  -s True > simulated_log.txt	
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
				Client can be in simulation mode - send auto generated messages
				Or in non simulation mode - User sends messages via terminal
		'''

		# Client gets an introductory message from server after connecting
		s_msg = self.socket.recv(1024).decode()
		print(f'{s_msg}')

		while True:
			# send a message TO server

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