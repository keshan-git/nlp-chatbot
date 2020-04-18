import numpy as np
import tensorflow as tf
import re
import time

lines = open('dataset/movie_lines.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
conversations = open('dataset/movie_conversations.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
