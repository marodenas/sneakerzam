import pandas as pd
from math import *


class Recommender:
    """
    Custom random recommender system based on pearson correlatoin. It will pick up similiar user
    wit same rating in sneakers, then multiply weight * rated sneakers and order the best punctuation
    in order to know recommended sneakers.
    """

    def __init__(self):
        usecols = ['userId', 'sneakerId', 'rating']
        self.dfsneakers = pd.read_csv("data/sneakers.csv")

        ratings = pd.read_csv('data/ratings_sneakersv1.csv', usecols=usecols)
        ratings = ratings.drop_duplicates()
        self.ratings_df = ratings

    def dic_transform(self, sneakers_selected, punctuations):
        input = [{'shoe': sneakers_selected[i], 'rating': punctuations[i]} for i in range(0, len(punctuations))]
        dfuser = pd.DataFrame(input)
        return dfuser

    def recommender_system(self, dict):
        inputSneakers = dict
        inputId = self.dfsneakers[self.dfsneakers['shoe'].isin(inputSneakers['shoe'].tolist())]
        # Then merging it so we can get the sneakerid. It's implicitly merging it by title.
        inputSneakers = pd.merge(inputId, inputSneakers)
        # Dropping information we won't use from the input dataframe
        inputSneakers = inputSneakers.drop('Unnamed: 0', 1)
        # Filtering out users that have rated sneakers that the input has rated and storing it

        userSubset = self.ratings_df[self.ratings_df['sneakerId'].isin(inputSneakers['sneakerId'].tolist())]
        userSubsetGroup = userSubset.groupby(['userId'])
        userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

        # Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
        pearsonCorrelationDict = {}

        # For every user group in our subset
        for name, group in userSubsetGroup:
            # Let's start by sorting the input and current user group so the values aren't mixed up later on
            group = group.sort_values(by='sneakerId')
            inputSneakers = inputSneakers.sort_values(by='sneakerId')
            # Get the N for the formula
            nRatings = len(group)
            # Get the review scores for the sneakers that they both have in common
            temp_df = inputSneakers[inputSneakers['sneakerId'].isin(group['sneakerId'].tolist())]
            # And then store them in a temporary buffer variable in a list format to facilitate future calculations
            tempRatingList = temp_df['rating'].tolist()
            # Let's also put the current user group reviews in a list format
            tempGroupList = group['rating'].tolist()
            # Now let's calculate the pearson correlation between two users, so called, x and y
            Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
            Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
            Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(
                tempGroupList) / float(nRatings)

            # If the denominator is different than zero, then divide, else, 0 correlation.
            if Sxx != 0 and Syy != 0:

                pearsonCorrelationDict[name] = abs(Sxy) / sqrt(abs(Sxx) * abs(Syy))

            else:
                pearsonCorrelationDict[name] = 0

        pearsonCorrelationDict.items()
        # clean df to have similarity rate in a column
        pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
        pearsonDF.columns = ['similarityIndex']
        pearsonDF['userId'] = pearsonDF.index
        pearsonDF.index = range(len(pearsonDF))
        # 50 top users by similarity rate
        topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
        # top users with sneakers that they rated
        topUsersRating = topUsers.merge(self.ratings_df, left_on='userId', right_on='userId', how='inner')
        # regla de negocio to give weight to users.
        topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']
        # Applies a sum to the topUsers after grouping it up by userId
        tempTopUsersRating = topUsersRating.groupby('sneakerId').sum()[['similarityIndex', 'weightedRating']]
        tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
        # Creates an empty dataframe
        recommendation_df = pd.DataFrame()
        # Now we take the weighted average
        recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / \
                                                                     tempTopUsersRating['sum_similarityIndex']

        recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
        rdf = recommendation_df.reset_index()
        rdf = rdf.sort_values(by='weighted average recommendation score', ascending=False).head(5)
        return rdf['sneakerId'].to_list()
        # return userSubsetGroup
