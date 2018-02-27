from mnb.naive_bayes_classifier import MultinomialNaiveBayesClassifier

if __name__ == '__main__':
    train_data = ['the most fun film of the summer',
                  'very powerful',
                  'just plain boring',
                  'entirely predictable and lacks energy',
                  'no surprises and very few laughs',
                  'average performance',
                  'average performance']

    train_target = [1, 1, 0, 0, 0, 2, 2]

    model = MultinomialNaiveBayesClassifier()
    model.fit(train_data, train_target)

    test_data = ['predictable with no fun',
                 'predictable with few fun',
                 'very very fun',
                 'very fun',
                 'average']

    test_target = [0, 0, 1, 0, 2]

    pred = model.predict(test_data)

    accuracy = sum(1 for i in range(len(pred)) if pred[i] == test_target[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))
