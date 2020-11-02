
def calculate_qer_score(features, labels):
    """
    When a feature has low score, the diference between the probabilities 
    for the positive and negative classes is high; therefore the feature is 
    more class specifc and more valuable for classifcation process.
    
    score(f) = p(f) + q(f)/ |p(f) - q(f)|
    p(f) = df(+) + .5/ n(+) + 1.0
    q(f) = df(-) +.5/ n(-) + .5
    df(+) --> document of f with positive class
    n(+) ---> total document with positive class
    """

    # create a dataset
    dataset = [*zip(features, labels)]

    # count frequency each class labels
    label_counts = Counter(train_labels)

    # extract all features
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(features)
    feature_names = count_vectorizer.get_feature_names()

    qer_score = {}
    for word in feature_names:
        doc_freq = {}

        # count frequency each word at certain class labels
        word = ' '+word+' '

        for label in label_counts:
            doc_freq[label] = 0

        for doc in dataset:
            sentence = ' '+doc[0]+' '
            if word in sentence:
                doc_freq[doc[1]] += 1

        pf = (doc_freq[1] + .5) / (label_counts[1] + 1.0)
        qf = (doc_freq[0] + .5) / (label_counts[0] + .5)
        score = (pf + qf) / abs(pf - qf)
        qer_score[word.strip()] = score
        
    return qer_score

