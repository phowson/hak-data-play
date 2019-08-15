import pandas as pd

def clean(x):
	return x.replace('"', '').strip();



contents = [];
with open('./ncbi.csv', 'r', encoding = 'utf-8') as f:
	contents = f.readlines()


dates = [];
journals = [];
identifiers = [];
titles = [];
bodies = [];

for l in contents[1:]:

	parts = l.split(',');

	end = 0;
	for k in range(len(parts)-1, 0, -1):
		if (len(parts[k].strip())>0) :
			end = k;
			break;

	abstractEnd = -1;
	for k in range(end-2, 0, -1):
		if "." in parts[k]:
			abstractEnd = k+1;
			break;

	if abstractEnd==-1:
		abstractEnd=end-1;


	date = clean(parts[end]);
	journal =clean(", ".join(parts[abstractEnd:end]));
	try: 
		identifier = int(clean(parts[0]));
		title= clean(parts[1]);
		rest = clean(",".join(parts[2:abstractEnd]));


		dates.append(date);
		journals.append(journal);
		identifiers.append(identifier);
		titles.append(title);
		bodies.append(rest);
	except Exception:
		print("Bad line : " ,parts);

print("Make pd dataframe");
df = pd.DataFrame(index=identifiers, data = { 'date' : dates,  'journal' : journals, 'title' : titles, 'body' : bodies});
print("done");
print(df.dtypes)
print(df.head());

df.to_pickle('./ncbi.pickle');
