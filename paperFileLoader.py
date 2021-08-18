path = "./dataset/paper"
file_list = os.listdir(path)
papers = []

for dataset in file_list:
    print(dataset)
    with open(path + "/" + dataset, 'r', encoding='utf-`8') as f:
        json_data = json.load(f)
        json_data = json_data['data']

        for doc in json_data:
            paper = doc['summary_entire'][0]['orginal_text']
            papers.append(paper)


# 너무 긴 문장 제거
filtered = filter(lambda x: len(x.split(' ')) < 300, papers)
papers = list(filtered)

series = pd.Series(papers)

len(series)

# Create a very small test set to compare generated text with the reality
trainset, testset = train_test_split(series, test_size=0.2)
print(len(trainset), len(testset))

for row in trainset:
    print("================")
    print(f"{trainset}")
    print(",,,,,,,,,,,,,,,")
    print(f"{row[:1000]}")
