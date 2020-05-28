try:
    import cupy as cp
except:
    import numpy as cp
import numpy as np
from datetime import datetime
from numba import njit
import random
import matplotlib.pyplot as plt

epsilon = 1e-5  # stability constant
dtypes = "void(int64[:,:], int64[:,:], int64[:], int64[:], int64, int64[:], int64)"

"""
This version on every epoch goes over every word instead of every relation in the closure.
"""


@njit(cache=True)
def sample_negs(positives, negatives, words, randoms, num_random, index, length):
    """
    This function randomly selects num_negs number of negatives for each of the given words in a batch

    :param positives: 2D array of size (batch_size, len(self.embedding)) that represents for each word, a binary array
     of 0 or 1 for whether or not the word represented by the index is a positive for that sample
     i.e. positives[i] = binary array of all samples where index indicates the sample word and 0/1 based on whether that
     sample is a positive for word words[i]
    :param negatives: 2D array of size (batch_size, num_negs) that represents for each word, an array of values that
    represent negative samples for the corresponding word in words.
    i.e. negatives[i] = negatives for word words[i]
    :param words: array of words we want to find negatives for
    :param randoms: buffer that holds random numbers
    :param num_random: number of random numbers to generate in case buffer runs out
    :param index: index that we are at in the buffer of random numbers
    :param length: number of total words in the vocab
    """
    for i in range(len(negatives)):
        for j in range(len(negatives[i]) - 1):
            rand_int = words[i]
            while rand_int == words[i] or positives[i, rand_int] == 1:
                if index[0] >= len(randoms):
                    randoms[:] = np.random.randint(0, length, num_random)
                    index[0] = 0
                rand_int = randoms[index[0]]
                index[0] = index[0] + 1
            negatives[i, j + 1] = rand_int


def compute_gradients_gpu(u, v, lr, batch_size, num_negs, dimensions, indices, negatives):
    """
    This function computes and applies the gradient updates for the batch os samples, applying them inside
    the function then returning the final values that the embedding should be set too.

    :param u: the embedding values for the words we are computing gradients for (batch_size, dimensions)
    :param v: the embedding values for the positive sample + negatives samples (1 + num_negs, dimensions)
    :param lr: learning rate
    :param batch_size: size of the current batch, may vary from regular batch_size because of last remaining samples
    that can't fit into an entire batch_size in an epoch
    :param num_negs: number of negative samples for each positive sample
    :param dimensions: dimensions of the embedding
    :param indices: the indices of all the u values (batch_size)
    :param negatives: the indices of all the v values (batch_size, 1 + num_negs)
    :return: Returns the values that the embedding of the samples and negatives should be set too as well as the loss

    Once again, calculation was adapted from R. Řehůřek and P. Sojka
    RaRe-Technologies: Software Framework for Topic Modelling with Large Corpora
    https://github.com/RaRe-Technologies/gensim
    """
    u = cp.asarray(u)
    v = cp.asarray(v)
    u_orig = u
    v_orig = v
    u = u.T[cp.newaxis, :, :]
    v = cp.swapaxes(cp.swapaxes(v, 1, 2), 0, 2)
    norm_u = cp.linalg.norm(u, axis=1)
    norm_v = cp.linalg.norm(v, axis=1)
    distances = cp.linalg.norm(u - v, axis=1)
    alpha = 1 - norm_u ** 2
    beta = 1 - norm_v ** 2
    gamma = 1 + 2 * ((distances ** 2) / (alpha * beta))
    poincare_distances = cp.arccosh(gamma)
    exp_negative_distances = cp.exp(-poincare_distances)
    sum = cp.sum(exp_negative_distances, axis=0)

    distances_squared = distances ** 2
    c = (4 / (alpha * beta * cp.sqrt(gamma ** 2 - 1)))[:, cp.newaxis, :]
    u_coeff = ((distances_squared + alpha) / alpha)[:, cp.newaxis, :]
    distance_gradient_u = (u_coeff * u - v)
    distance_gradient_u *= c

    v_coeffs = ((distances_squared + beta) / beta)[:, cp.newaxis, :]
    distance_gradients_v = (v_coeffs * v - u)
    distance_gradients_v *= c

    gradients_v = -exp_negative_distances[:, cp.newaxis, :] * distance_gradients_v
    gradients_v /= sum
    gradients_v[0] += distance_gradients_v[0]
    gradients_v[0] += lr * 2 * v[0]

    gradients_u = -exp_negative_distances[:, cp.newaxis, :] * distance_gradient_u
    gradients_u /= sum
    gradient_u = cp.sum(gradients_u, axis=0)
    gradient_u += distance_gradient_u[0]

    u_update = (lr * (alpha ** 2) / 4 * gradient_u).T
    handle_duplicates(u_update, indices)
    v_updates = cp.swapaxes(cp.swapaxes((lr * (beta ** 2)[:, cp.newaxis, :] / 4 * gradients_v), 0, 2), 1, 2)
    v_updates = cp.reshape(v_updates, (batch_size * (num_negs + 1), dimensions))
    handle_duplicates(v_updates, np.ravel(negatives))

    u_orig -= u_update
    v_orig = cp.reshape(v_orig, (batch_size * (num_negs + 1), dimensions))
    v_orig -= v_updates
    u_norms = cp.linalg.norm(u_orig, axis=1)
    v_norms = cp.linalg.norm(v_orig, axis=1)

    u_orig = (u_norms >= 1 - epsilon)[:, cp.newaxis] * (u_orig / u_norms[:, cp.newaxis] - cp.sign(u_orig) * 0.00001) + (u_norms < 1 - epsilon)[:,cp.newaxis] * u_orig

    v_orig = (v_norms >= 1 - epsilon)[:, cp.newaxis] * (v_orig / v_norms[:, cp.newaxis] - cp.sign(v_orig) * 0.00001) + (v_norms < 1 - epsilon)[:,cp.newaxis] * v_orig

    loss = cp.sum(-cp.log(exp_negative_distances[0] / sum), axis=0)
    return u_orig, v_orig, loss


def compute_gradients_cpu(u, v, lr, batch_size, num_negs, dimensions, indices, negatives):
    """
        This function computes and applies the gradient updates for the batch os samples, applying them inside
        the function then returning the final values that the embedding should be set too.

        :param u: the embedding values for the words we are computing gradients for (batch_size, dimensions)
        :param v: the embedding values for the positive sample + negatives samples (1 + num_negs, dimensions)
        :param lr: learning rate
        :param batch_size: size of the current batch, may vary from regular batch_size because of last remaining samples
        that can't fit into an entire batch_size in an epoch
        :param num_negs: number of negative samples for each positive sample
        :param dimensions: dimensions of the embedding
        :param indices: the indices of all the u values (batch_size)
        :param negatives: the indices of all the v values (batch_size, 1 + num_negs)
        :return: Returns the values that the embedding of the samples and negatives should be set too as well as the loss

        Once again, calculation was adapted from R. Řehůřek and P. Sojka
        RaRe-Technologies: Software Framework for Topic Modelling with Large Corpora
        https://github.com/RaRe-Technologies/gensim
        """
    u_orig = u
    v_orig = v
    u = u.T[np.newaxis, :, :]
    v = np.swapaxes(np.swapaxes(v, 1, 2), 0, 2)
    norm_u = np.linalg.norm(u, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    distances = np.linalg.norm(u - v, axis=1)
    alpha = 1 - norm_u ** 2
    beta = 1 - norm_v ** 2
    gamma = 1 + 2 * ((distances ** 2) / (alpha * beta))
    poincare_distances = np.arccosh(gamma)
    exp_negative_distances = np.exp(-poincare_distances)
    sum = np.sum(exp_negative_distances, axis=0)

    distances_squared = distances ** 2
    c = (4 / (alpha * beta * np.sqrt(gamma ** 2 - 1)))[:, np.newaxis, :]
    u_coeff = ((distances_squared + alpha) / alpha)[:, np.newaxis, :]
    distance_gradient_u = (u_coeff * u - v) * c

    v_coeffs = ((distances_squared + beta) / beta)[:, np.newaxis, :]
    distance_gradients_v = (v_coeffs * v - u) * c

    gradients_v = -exp_negative_distances[:, np.newaxis, :] * distance_gradients_v
    gradients_v /= sum
    gradients_v[0] += distance_gradients_v[0]

    gradients_u = -exp_negative_distances[:, np.newaxis, :] * distance_gradient_u
    gradients_u /= sum
    gradient_u = np.sum(gradients_u, axis=0) + distance_gradient_u[0]

    u_update = (lr * (alpha ** 2) / 4 * gradient_u).T
    handle_duplicates(u_update, indices)
    v_updates = np.swapaxes(np.swapaxes((lr * (beta ** 2)[:, np.newaxis, :] / 4 * gradients_v), 0, 2), 1, 2)
    v_updates = np.reshape(v_updates, (batch_size * (num_negs + 1), dimensions))
    handle_duplicates(v_updates, np.ravel(negatives))

    u_orig -= u_update
    v_orig = np.reshape(v_orig, (batch_size * (num_negs + 1), dimensions))
    v_orig -= v_updates
    u_norms = np.linalg.norm(u_orig, axis=1)
    v_norms = np.linalg.norm(v_orig, axis=1)

    u_orig = (u_norms >= 1 - epsilon)[:, np.newaxis] * (u_orig / u_norms[:, np.newaxis] - np.sign(u_orig) * 0.00001) + (u_norms < 1 - epsilon)[:,np.newaxis] * u_orig

    v_orig = (v_norms >= 1 - epsilon)[:, np.newaxis] * (v_orig / v_norms[:, np.newaxis] - np.sign(v_orig) * 0.00001) + (v_norms < 1 - epsilon)[:,np.newaxis] * v_orig

    loss = np.sum(-np.log(exp_negative_distances[0] / sum), axis=0)
    return u_orig, v_orig, loss


def handle_duplicates(updates, idxs):
    """
    Handles duplicate updates by summing the updates and setting the last of the duplicates
    to the sum, while zero-ing out the other updates.

    :param updates: update values that will be applied
    :param idxs: indices corresponding to the update values
    :return:
    """
    positions = dict()
    seen = set()
    for i, n in enumerate(idxs):
        if n in seen:
            if n not in positions:
                positions[n] = list()
            positions[n].append(i)
        else:
            seen.add(n)
    for n in positions:
        px = positions[n]
        updates[px[-1]] = np.sum(updates[px], axis=0)
        updates[px[:-1]] = 0


class PoincareEmbedding(object):
    """
    This class generates an embedding (collection of positions in hyperbolic space with
    the given number of dimensions). Initialize it with a transitive closure which should be
    a simple list of tuples (u, v) which represents a directed relationship from u -> v.
    Implementation based on:
    https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations.pdf
    Full credit for gradient calculation to R. Řehůřek and P. Sojka
    RaRe-Technologies: Software Framework for Topic Modelling with Large Corpora
    https://github.com/RaRe-Technologies/gensim

    Usage:
    model = PoincareEmbedding(params)
    model.fit_transform(epochs)
    embedding = model.embedding

    """

    def __init__(self, closure, dimensions=2, num_negs=10, lr=0.1, burn_in=False, batch_size=1000, num_random=1000000,
                 print=True, gpu=False):
        """
        Initializes the embedding's data structures and random buffer.

        :param closure: List of relations in the form of [(a, b), (c, d), (e, f)...] where each tuple represents
        a one-way relation from the first element to the second element. i.e. (a, b) = a -> b
        :param dimensions: Dimensions of the embedding
        :param num_negs: Number of negatives we want to sample for each sample
        :param lr: Learning rate
        :param burn_in: Burn in runs 10 epochs at a learning rate of lr / 10 to help the model get better initial positions
        :param batch_size: Size of each batch os samples
        :param print: Whether or not to print time and loss information
        :param gpu: Whether of not to use GPU (CuPy)

        After pre-processing, words are represented by integers. word2index and vocab (index to word) help inter-convert
        """
        self.dimensions = dimensions
        self.num_negs = num_negs
        self.lr = lr
        self.closure = closure
        self.batch_size = batch_size
        self.num_random = num_random
        self.print = print
        self.gpu = gpu

        self.word2positives, self.vocab, self.word2index = self.__initialize()

        self.embedding = np.random.uniform(low=-0.001, high=0.001, size=(len(self.vocab), dimensions))
        self.length = len(self.embedding)

        self.randoms = np.random.randint(0, self.length, size=num_random)  # Random number buffer
        self.index = np.zeros(1, dtype=int)

        # These data structures are initialized as fields so we can re-use them instead of making new ones
        # each epoch. 
        # negatives is the indices of the positive sample + negatives samples for each sample in a batch
        # (batch_size, num_negs + 1)
        # positives is a binary array for the positives of each sample in a batch
        # (batch_size, len(embedding))
        # batch_idxs are the indices for the current batch of samples we are looking to update
        # (batch_size)
        # new_indices is a shuffled list of numbers from 0 to len(embedding) so that we can randomly choose
        # different samples to update each epoch
        # (len(embedding))
        # indices is the actual selection of batch_size samples that we are looking to update
        # (batch_size)
        self.new_indices = np.random.permutation(len(self.embedding))

        if burn_in:
            self.__burn_in()

    def __burn_in(self):
        """
        Burns in by training for 10 epochs using a regularization rate of lr / 10
        """
        if self.print:
            print("Burning in...")
        for epoch in range(0, 10):
            np.random.shuffle(self.new_indices)
            self.__fit_transform(self.lr / 10)
        if self.print:
            print("Done burning in!")

    def __initialize(self):
        """
        Initializes the data structure to be used in training

        :return:
        word2positives: is a dict from word (str) -> list of integer values, each integer represents a word
        vocab: is the complete list of words, where the index corresponds to the integer value of a word
        word2index: is a dict that allows us to get the integer value of a string word
        """
        word2positives = dict()
        vocab = list()
        vocab_set = set()
        word2index = dict()
        for (u, v) in self.closure:
            if u not in vocab_set:
                vocab_set.add(u)
                vocab.append(u)
                word2index[u] = len(vocab) - 1
            if v not in vocab_set:
                vocab_set.add(v)
                vocab.append(v)
                word2index[v] = len(vocab) - 1
            u_num = word2index[u]
            v_num = word2index[v]
            if u_num not in word2positives:
                word2positives[u_num] = list()
            if v_num not in word2positives:
                word2positives[v_num] = list()
            word2positives[u_num].append(v_num)
            word2positives[v_num].append(u_num)
        return word2positives, vocab, word2index

    # Trains with the given number of epochs
    def fit_transform(self, epochs):
        """
        Fits the PoincareEmbedding with the given number of epochs to the closure

        :param epochs: Number of epochs
        """
        for epoch in range(1, epochs + 1):
            np.random.shuffle(self.new_indices)
            if self.print:
                print("Training Epoch: " + str(epoch))
            self.__fit_transform(self.lr)

    def __fit_transform(self, lr):
        """
        This function first samples negatives into the negatives array and places the correct information into
        the other arrays, then passes the arrays to the computation function. The computation function
        calculates and applies the updates, and returns the updates as a list that is then set inside the class.

        :param lr: Learning rate to train at
        """
        total_loss = 0.0
        start_time = datetime.now()
        index = 0
        while index < len(self.embedding):
            indices = []
            while index < len(self.embedding) and len(indices) < self.batch_size:
                if (len(self.word2positives[self.new_indices[index]]) != 0) \
                        and (len(self.word2positives[self.new_indices[index]]) != len(self.embedding) - 1):
                    indices.append(self.new_indices[index])
                index += 1

            if len(indices) == 0:
                break

            positives = np.zeros((len(indices), len(self.embedding)), dtype=int)
            negatives = np.zeros((len(indices), self.num_negs + 1), dtype=int)
            for idx, val in enumerate(indices):
                word1 = val
                word2 = random.choice(self.word2positives[val])
                for positive in self.word2positives[word1]:
                    positives[idx, positive] = 1
                negatives[idx, 0] = word2

            sample_negs(positives, negatives, indices, self.randoms, self.num_random, self.index, self.length)

            if self.gpu:
                u_updates, v_updates, loss = compute_gradients_gpu(self.embedding[indices],
                                                                   self.embedding[negatives],
                                                                   lr, len(indices), self.num_negs, self.dimensions, indices, negatives)
                u_updates = cp.asnumpy(u_updates)
                v_updates = cp.asnumpy(v_updates)
            else:
                u_updates, v_updates, loss = compute_gradients_cpu(self.embedding[indices],
                                                                   self.embedding[negatives],
                                                                   lr, len(indices), self.num_negs, self.dimensions, indices, negatives)
            self.embedding[indices] = u_updates
            self.embedding[np.ravel(negatives)] = v_updates
            total_loss += loss
        end_time = datetime.now()
        delta = end_time - start_time
        if self.print:
            print("Total Loss: " + str(total_loss))
        if self.print:
            print("Time : " + str(delta))
