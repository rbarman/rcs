'''
AgentServer.py

AgentServer is a server socket that accepts connections from multiple client sockets.
Agent's represent the authority figures and analyze messages between clients
Agent's must pass in an auth_config. The config creates constratins to 
	determine what messages are radical or not. 

Example usage:
	python AgentServer.py -p 9999 -c ~/Desktop/auth_config.txt
'''

import socket
import argparse
import select
import json
import random
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import  word_tokenize
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

class AgentServer:

	def __init__(self,port,config, max_connection_count = 10):
		# bind the server socket on localhost + specified port
		self.port = port
		self.socket = socket.socket()
		self.socket.bind(('localhost',self.port))
		# max number of connections server will listen to
		self.socket.listen(max_connection_count)
		self.connection_list = []
		self.connection_list.append(self.socket)

		# convert config to pandas dataframe
		self.config = config
		self.config_df = pd.read_csv(self.config)

		# store dictionary of each user
		self.user_dict = {}
		self.user_check_count = {}

		print('----------------------------------')
		print(f'Config:')
		print(self.config_df)
		print(f'Listening to connections on {port}')
		print('----------------------------------')

	def get_sentiment_scores(self,sentence):
		sid = SentimentIntensityAnalyzer()
		scores = sid.polarity_scores(sentence)
		return scores

	def auth_keyword_search(self,sentence):
		''' simple intersection to find keyword matches'''
		auth_set = set(self.config_df.Keyword.values)
		# tokenize word to account for punctuation in string
		tokenized = (word_tokenize(sentence.lower()))
		sent_set = set(tokenized)
		found_keywords = list(auth_set.intersection(sent_set))
		return found_keywords	

	def upate_user_dict(self,user,sentence):
		# sentiment scores
		scores = self.get_sentiment_scores(sentence)
		# key word search
		matched_keywords = self.auth_keyword_search(sentence)

		for keyword in matched_keywords:
			self.user_dict[f'{user}'][f'{keyword}'].append(scores)

	def calc_score_metrics(self,user_id,keyword,stance, most_recent=3):
		
		scores = []
		for record in self.user_dict[f'{user_id}'][f'{keyword}']:
			scores.append(record[f'{stance}'])
			
		mean = round(np.mean(scores),2)
		std = round(np.std(scores),2)
		mean_recent = round(np.mean(scores[-1*most_recent:]),2)
	
		return mean,std,mean_recent

	def run(self):
		while True:
			''' Server is running, accepting connections + messages from clients, and sending back messages'''

			# select returns sockets where we can read to, write to, or have errors
				# for now we only care about sockets we can read from 
			read_sockets,_,_ = select.select(self.connection_list,[],[])
			
			# The notified socket can either be
				# 1) Server Socket - received a new client connection
				# 2) Another socket - Client socket is sending a message to the server
			for notified_socket in read_sockets:

				# Established connecton with a new client
					# Send an introductory message to the Client
				if notified_socket == self.socket:
					c, c_address = notified_socket.accept()
					print(f'connected to client from {c_address}')				
					self.connection_list.append(c)
					connected_peer = c_address[1]
					#create a new entry with the client's peer name
					self.user_dict[f'{connected_peer}'] = {}
					# create an empty list for each config keyword
					for k in self.config_df.Keyword.values:
						self.user_dict[f'{connected_peer}'][f'{k}'] = []

					self.user_check_count[f'{connected_peer}'] = 0

					# send introduction message
					c.send(bytes('Connected','utf-8'))
					
				# Client socket sent server a json its GSPs + message
					# send back a response to client based on GSPs and sentiment of message	
				else:
					c_message = notified_socket.recv(1024).decode() 
					connected_peer = notified_socket.getpeername()[1]
					# print(f'{connected_peer} : {c_message}')

					# process client's message
						# update user's message history
					self.upate_user_dict(connected_peer,c_message)
					self.user_check_count[f'{connected_peer}']  += 1

					# check on the user history every n times
					if self.user_check_count[f'{connected_peer}'] %10 == 0:
						for keyword, radical_stance, threshold in zip(self.config_df.Keyword,self.config_df.Radical_stance,self.config_df.Accepted_threshold):
							mean,std,mean_recent = self.calc_score_metrics(connected_peer,keyword,radical_stance)
							if mean > threshold:
								print(f'User {connected_peer} is spreading radical info about {keyword}')
								print(f'\t{radical_stance} scores=> mean={mean}\tstd={std}\tmean_recent={mean_recent}')
								if abs(std) < .1:
									print('\tpossible that this user is already radicalized')
								if mean_recent > mean:
									print('\tpossible that this user is beginning to get radicalized')
								if mean_recent < mean:
									print('\tpossible that this user is in the process of getting deradicalized')
							
								# for now pick a random connection for connected peer
								messaged_peer = random.choice(list(self.user_dict.keys()))
								if messaged_peer != connected_peer:
									messaged_mean,messaged_std,messaged_mean_recent = self.calc_score_metrics(messaged_peer,keyword,radical_stance)

									print(f'\t{connected_peer} messaged with {messaged_peer}')
									if messaged_mean > mean:
										print(f'\t\t{messaged_peer} has been spreading more randical content about {keyword} - Make sure that {messaged_peer}:{connected_peer} connection does not promote more radical ideas')
									else:
										if messaged_mean < threshold:
											print(f'\t\t{messaged_peer} has not spread any radical ideas... {connected_peer}:{messaged_peer} connection could radicalize {messaged_peer}')
										else:
											print(f'\t\t{messaged_peer} has not spread as much radical ideas... {connected_peer}:{messaged_peer} connection further radicalize {messaged_peer}')

							else:
								if mean_recent > mean:
									print(f'User {connected_peer} not sharing but possibly getting radicalized about {keyword}')
									print(f'\t{radical_stance} scores => mean={mean}\tstd={std}\tmean_recent={mean_recent}')
if __name__ == '__main__':

	# set up arg parser
	parser = argparse.ArgumentParser(description='Agent Server')
	parser.add_argument('-c','--config',help='CSV file of keyword, radical stance, and accepted threshold')
	parser.add_argument('-p','--port',help='Port for server to listen to ', type=int, default=9572)
	args = parser.parse_args()
	
	# Start server based on localhost + user input ports
	server = AgentServer(args.port,args.config)
	server.run()