
# coding: utf-8

# In[31]:


from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import json
import itertools


# In[32]:


consumer_key = 'NGcCfRJnLeGljgyzpLwDRi6qT'
consumer_secret = 'OzDJ7fTb8hTjN5HJ5Y4AYnRSvLSww8QDzpM9gq2Ox54JDvluqd'
access_token = '1183048256-3mFMGnCFzUkgWegCamZSnI9zctObpl8WfZfPxZL'
access_token_secret = 'XrqZnYQcsoUIhyM4vS5etG3wHO3644QgUstNcSwJSDnLB'


# In[33]:


def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


# In[34]:


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.
    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.
    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    ###TODO
    
    
    candidates = open(filename) #simply storing the file in candidates first and then splitting the names, and returning. 
    names = candidates.read().split()
    return names
    
    


# In[35]:


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=15):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


# In[36]:


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)
    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup
    In this example, I test retrieving two users: twitterapi and twitter.
    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    twitter_data_api=robust_request(twitter,"users/lookup", {'screen_name': screen_names})   #Twitter API call.
    candidate_data=[]                                   #Simple storing, getting api data and appending in the data structure. 
    for user in twitter_data_api:
        candidate_data.append(user)
    return candidate_data
    
    


# In[37]:


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids
    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.
    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.
    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.
    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    twitter_data_api = robust_request(twitter,'friends/ids',{'screen_name' : screen_name})         #twitter api call.
    friends=[]                                              #same as getting users and sorting is done here.
    for ID in twitter_data_api:
            friends.append(ID)
    return (sorted(friends))
    
    


# In[38]:


def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.
    Store the result in each user's dict using a new key called 'friends'.
    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing
    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    for user in users:
        username = user['screen_name']
        user['friends']= get_friends(twitter, username)
    
    


# In[39]:


def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    ###TODO
    for u in users:
        num_friends=str(len(u['friends']))
        print((u['screen_name'])+ "  " +num_friends)
        
    


# In[40]:


def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter
    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    ###TODO
    numfriends=[]
    
    for user in users:
        numfriends = numfriends + user['friends']
        
    count = Counter(numfriends)
    return count
    
    
    


# In[41]:


def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.
    Args:
        users...The list of user dicts.
    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.
    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    ###TODO
    candidates = []
    name_dict = {}

    for user in users:
        candidates.append(user['screen_name'])

    notations = list(itertools.combinations(candidates,2))

    for name_values in notations:
        for user in users:
            if name_values[0] == user['screen_name'] or name_values[1] == user['screen_name']:
                if name_values not in name_dict:
                    name_dict[name_values] = user['friends']
                else:
                    name_dict[name_values] = set(user['friends']).intersection(name_dict[name_values])

    overlap_names = []
    for name_values in notations:
        overlap_names.append((name_values[0],name_values[1],len(name_dict[name_values])))

    result = sorted(tuple(overlap_names), key=lambda t: (-t[2],t[0],t[1]))   #the '-' before t[2] does the descending sort, if removed it sorts in ascending order. 

    return result


# In[42]:


def followed_by_hillary_and_donald(users, twitter):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup
    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    ###TODO
    hillary = get_friends(twitter,'HillaryClinton')
    trump = get_friends(twitter,'realDonaldTrump')
    both = set(hillary).intersection(trump)
    result = 'No one followed by both Hillary and Donald'

    if len(both) >= 1:
        a = both.pop()
        twitter_api_data = robust_request(twitter,"users/lookup",{'user_id': a})
        for item in twitter_api_data.get_iterator():
                result = item['screen_name']

    return result
    


# In[43]:


def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)
        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.
    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    ###TODO
    
    graph = nx.Graph()
    for user in users:
        graph.add_node(user['screen_name'])
        for frnd in user['friends']:
            if friend_counts[frnd] > 1:
                graph.add_edge(frnd, user['screen_name'])
                
    return graph
    


# In[44]:


def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.
    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """ 
    ###TODO
    notation = dict()
    for u in users:
        notation[u['screen_name']] = u['screen_name'] 
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10,10))
    #nx.draw_networkx(graph, pos, arrows=True, labels=labels, with_label=True, ax=None, nodelist = graph.nodes(), edgelist=graph.edges(), node_size=100, alpha=0.5, width= 0.5, font_size=10)
    nx.draw_networkx_labels(graph, labels=notation, pos=pos,font_size=10)           #labels= name of the label to reduce clutter, else it includes the id of every node except the candidates.
    nx.draw_networkx_nodes(graph, pos, alpha=0.5, node_size=55, node_color='red')
    nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.15)
    plt.axis('off')
    plt.savefig(filename)
    


# In[45]:


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
            main()


# In[ ]:





# In[ ]:





# In[ ]:




