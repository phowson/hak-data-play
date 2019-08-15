
import pickle
from hierarchicalclusterplay1 import NavHierarchyTopic, Ticket
import sys;
import re;

def clean(b):
	b = re.sub( '[^A-z 0-9]',' ', b);
	return b;


def writeJson(fls, topic):

	print("{", file = fls);
	str = "root";

	if topic.navTopicTuple!=None:		
		str = ",".join(map( lambda x: x[0] , topic.navTopicTuple[1]));	
		print('	"collapsed"  : "true",', file = fls);
	else:
		print('	"collapsed"  : "false",', file = fls);
	print('	"topicName" : "' + str +'", ', file = fls);	
	print('	"name" : "' + str +'"  , ', file = fls);


	titles = [];
	for x in topic.documentsByIndex.values():
		titles.append('"' + x.title  + '"');

	print('	"titles" : [' + ",".join(titles) +"], ", file = fls);
	print('	"_children" : ', file = fls);

	i = 0;

	for c in topic.children:
		if (i==0) :
			print("[", file = fls);
		else:
			print('	,', file = fls);


		writeJson(fls, c);
		i=i+1;
		
	if (i==0):
		print("[", file = fls);
		print(",".join(map(lambda x : '{ "name" : "' + clean(x) + '" }', titles)), file = fls);		 	
		print("	]", file = fls);

	else:	
		print("	]", file = fls);


	print("}", file = fls);



with open('cluster_hierarchy.pickle', 'rb') as f:
	o = pickle.load(f)


with open('tree.json', 'w', encoding = 'utf-8') as f:
	writeJson(f, o[0])
