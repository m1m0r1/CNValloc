# -*- coding: utf-8 -*-
import numpy as np
import logging
from itertools import groupby, product
from scipy.misc import logsumexp
from scipy.special import gammaln, digamma, polygamma


def log_normalize(log_vec, axis=0):
    axes = [slice(None)] * len(log_vec.shape)
    axes[axis] = np.newaxis
    return log_vec - logsumexp(log_vec, axis=axis)[axes]


def _calc_gamma_psi(log_d, alpha, log_beta, gamma, log_psi0):
    log_psi = log_psi0
    count = 0
    #print ((np.exp(log_psi1 - log_psi) ** 2).sum())

    while count == 0 \
        or ((np.exp(log_psi1) - np.exp(log_psi)) ** 2).sum() > 0.001 \
        or ((gamma1 - gamma) ** 2).sum() > 0.001:
        #print ('gamma psi:', count, ((np.exp(log_psi1) - np.exp(log_psi)) ** 2).sum())
        log_psi1 = log_psi
        gamma1 = gamma

        psi_offset = (digamma(gamma))[:, np.newaxis, np.newaxis, :]

        log_psi = log_beta[np.newaxis, :, :, :] + psi_offset
        log_psi = log_normalize(log_psi, axis=3)
        gamma = np.exp(logsumexp(logsumexp(log_d[:, :, :, np.newaxis] + log_psi, axis=1), axis=1)) + alpha[np.newaxis, :]
        count += 1

    #log_psi = np.average([log_psi0, log_psi], axis=0, weights=[0.9, 0.1])   # weak learning
    return (gamma, log_psi)


def _calc_alpha_beta(log_d, alpha0, log_beta0, gamma, log_psi):
    log_beta = logsumexp(log_psi + log_d[:, :, :, np.newaxis], axis=0)
    log_beta = log_normalize(log_beta, axis=1)

    log_smooth = np.log(10)
    alpha = alpha0
    N = gamma.shape[0]
    zero = 1e-30

    gamma_digamma_sum = digamma(gamma.sum(axis=1))[:, np.newaxis]
    g_offset = (digamma(gamma) - gamma_digamma_sum).sum(axis=0) / N
    # using log
    def next_alpha(alpha):
        das = digamma(alpha.sum())
        g = alpha * N * (das - digamma(alpha) + g_offset)
        h = alpha * N * (das + g_offset)
        z = N * das
        x = (alpha * g / h).sum()
        w = (alpha ** 2 / h).sum()
        return np.exp(np.log(alpha) - (g - x * alpha / (1/z + w)) / h)

    return (alpha, log_beta)


class LDAStatus(object):
    @classmethod
    def init_with_data(cls, lda_data, nhaps=3, haplotypes=None, use_ll_offset=False):
        V = len(lda_data.alphabets)

        status = cls(lda_data)

        zero = 1e-30  # value for zero count

        status.nhaps = nhaps
        status.ll_offset = gammaln(nhaps + 1) if use_ll_offset else 0.
        status.alpha = np.array([1. for _ in xrange(nhaps)])    # Uniform distribution
        #status.alpha = np.array([1. / nhaps for _ in xrange(nhaps)])
        if haplotypes:
            status.log_beta = np.log(np.array([[[1 if a == haplotypes[n][m] else zero for n in xrange(nhaps)] for a in lda_data.alphabets] for m in xrange(lda_data.nsites)]))
        else:
            status.log_beta = np.array([[[np.random.random() for _ in xrange(nhaps)] for _ in lda_data.alphabets] for _ in xrange(lda_data.nsites)])
        status.log_beta = log_normalize(status.log_beta, axis=1)
        # adhoc initialization
        status.log_psi = status.log_beta[np.newaxis, :, :, :]
        status.gamma = status.alpha[np.newaxis, :] + np.exp(lda_data.log_d[:, :, :, np.newaxis] + status.log_psi).sum(axis=1).sum(axis=1)
        status.ll = status.calc_lower_ll()

        #(status.gamma, status.log_psi) = _calc_gamma_psi(log_d, status.alpha, status.log_beta, gamma, log_psi)
        return status

    def __init__(self, lda_data, step=0):
        self._data = lda_data
        self.step = step
        self.ll = 0

    def with_new_gamma_psi(self):
        status = self.__class__(self._data, step=self.step)
        status.alpha = self.alpha
        status.log_beta = self.log_beta
        status.nhaps = self.nhaps
        status.ll_offset = self.ll_offset
        (status.gamma, status.log_psi) = _calc_gamma_psi(self._data.log_d, self.alpha, self.log_beta, self.gamma, self.log_psi)
        return status

    def with_new_alpha_beta(self):
        status = self.__class__(self._data, step=self.step)
        status.gamma = self.gamma
        status.log_psi = self.log_psi
        status.nhaps = self.nhaps
        status.ll_offset = self.ll_offset
        (status.alpha, status.log_beta) = _calc_alpha_beta(self._data.log_d, self.alpha, self.log_beta, self.gamma, self.log_psi)
        return status

    def get_next(self):
        status = self.__class__(self._data, step=self.step + 1)
        status.nhaps = self.nhaps
        status.alpha = self.alpha
        status.log_beta = self.log_beta
        status.gamma = self.gamma
        status.log_psi = self.log_psi
        status.ll_offset = self.ll_offset
        status = status.with_new_gamma_psi()
        status = status.with_new_alpha_beta()
        status.ll = status.calc_lower_ll()
        return status

    def calc_lower_ll(self):
        g = digamma(self.gamma) - digamma(self.gamma.sum(axis=1)[:, np.newaxis])
        N = self._data.nsamples
        psi_d = np.exp(self.log_psi + self._data.log_d[:, :, :, np.newaxis])

        ll = self.ll_offset
        ll += (psi_d.sum(axis=0) * self.log_beta).sum()
        ll += (psi_d.sum(axis=1).sum(axis=1) * g).sum()
        ll += N * (gammaln(self.alpha.sum()) - gammaln(self.alpha).sum()) + (self.alpha * g.sum(axis=0)).sum()
        ll -= (gammaln(self.gamma.sum(axis=1)) - gammaln(self.gamma).sum(axis=1) + (self.gamma * g).sum(axis=1)).sum()
        ll -= (psi_d * self.log_psi).sum()
        return ll

    def __str__(self):
        haps = np.array([[self._data.alphabets[np.argmax(site_beta)] for site_beta in beta_sites.transpose()] for beta_sites in self.log_beta]).transpose()

        sts = []
        sts.append('* step {0}'.format(self.step))
        sts.append('    alpha: {0}'.format(self.alpha))
        sts.append('haplotypes:')
        sts.append('\n'.join(
                   '         {0:d}: {1}'.format(i, ''.join(hap)) for (i, hap) in enumerate(haps)))
        mean_depths = np.exp(self._data.log_d).sum(axis=2).mean(axis=1)[:, np.newaxis]
        sts.append(' hap ratio: {0}'.format((mean_depths * self.gamma / self.gamma.sum(axis=1)[:, np.newaxis]).round(1)))
        sts.append('        ll: {0}'.format(self.ll))
        return '\n'.join(sts)

    def as_dict(self):
        """ Returns json encodable dictionary
        """
        haps = [[self._data.alphabets[np.argmax(site_beta)] for site_beta in beta_sites.transpose()] for beta_sites in self.log_beta]
        hap_probs = [[[np.exp(np.max(site_beta) - logsumexp(site_beta)) for site_beta in beta_sites.transpose()] for beta_sites in self.log_beta]]

        return {
            'nsites': self._data.nsites,
            'sample_ids': self._data.sample_ids,
            'nhaps': self.nhaps,
            'alphabets': list(self._data.alphabets),
            'step': self.step,
            'log_likelihood': self.ll,
            'alpha': list(map(float, self.alpha)),
            'log_beta': [[list(map(float, site_beta)) for site_beta in beta_sites.transpose()] for beta_sites in self.log_beta],
            'haps': haps,
            'hap_probs': hap_probs,
            'gamma': list(map(float, vec) for vec in self.gamma),
        }


class LDAData(object):
    alphabets = ['A', 'T', 'C', 'G']

    def __init__(self, histfile):
        def load_histfile(histfile):
            hists = []
            sample_ids = []

            def iter_rows():
                for line in open(histfile):
                    row = line.rstrip('\r\n').split('\t')
                    yield row

            # load histogram
            for (sample_id, rows) in groupby(iter_rows(), lambda x: x[0]):
                base_counts = {}

                for row in rows:
                    base = row[1]
                    counts = map(int, row[2:])
                    base_counts[base] = counts

                hists.append(base_counts)
                sample_ids.append(sample_id)
            return (sample_ids, hists)

        (self.sample_ids, self.hists) = load_histfile(histfile)   # [[{base: count}]]  # N x M x V
        #print (self.hists)
        self.nsamples = len(self.hists)        # number of samples
        self.nsites = len(self.hists[0]['A'])  # number of segregating sites
        logging.info('nsamples: %s', self.nsamples)
        logging.info('nsites: %s', self.nsites)
        logging.info('nsamples in hists: %s', len(self.hists))
        nbases = len(self.hists[0])
        nsites = len(self.hists[0]['A'])
        logging.info('bases in hists: %s', nbases)
        logging.info('sites in hists: %s', nsites)

        for i in xrange(self.nsamples):
            if len(self.hists[i]['A']) != nsites:
                raise Exception('number of sites are incompatible! ({0} != {1})'.format(len(self.hists[i]['A']), nsites))

        zero = 1e-30  # value for zero count
        self.log_d = np.log(np.array(
            [[[self.hists[n][base][x] or zero
                                for base in self.alphabets]
                                for x in xrange(self.nsites)]
                                for n in xrange(self.nsamples)]))
