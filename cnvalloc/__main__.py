#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import os, sys
if __package__ == '':   # for relative import
     (search_path, __package__) = os.path.split(os.path.split(os.path.abspath(__file__))[0])
     sys.path.insert(0, search_path)
     __import__(__package__)
import logging
from argtools import command, argument
from itertools import groupby, product
from collections import defaultdict
from math import factorial
import re
import vcf
try:
    from itertools import combinations_with_replacement
except ImportError:  # for python 2.6
    #from mcnv import combinations_with_replacement
    def combinations_with_replacement(iterable, r):
        """
        >>> list(combinations_with_replacement('AT', 2))
        [('A', 'A'), ('A', 'T'), ('T', 'T')]
        """
        assert isinstance(r, (int, long))
        assert r >= 0

        if r == 0:
            yield ()
        else:
            alls = list(iterable)
            for (i, el) in enumerate(alls):
                for els in combinations_with_replacement(alls[i:], r - 1):
                    yield (el,) + els
import json
import numpy as np
import pysam
from . import parse_region
from . import estimator


@command.add_sub
@argument('vcf')
@argument('-r', '--region', required=True, help='X:start-end')
@argument('--snp-only', action='store_true', default=False)
@argument('--min-maf', default=0, type=float)
@argument('--with-weight', action='store_true', default=False)
def vcf2haps(args):
    """
    Convert VCF file to haplotype info at variant sites

    Emit tsv file with following fields:
        1. hap_id
        2. comma separated hap bases

    If with_weight were set, emit
        1. hap_id
        2. comma separated hap bases
        3. weight
    """
    chrom = None
    start = 0
    end = None
    if args.region:
        (chrom, se) = args.region.split(':')
        start = int(se.split('-')[0])
        end = int(se.split('-')[1])

    reader = vcf.Reader(filename=args.vcf)
    for (chrom1, chr_reader) in groupby(reader, lambda rec: rec.CHROM):
        if chrom and chrom != chrom1:
            continue

        splitter = re.compile('/|\|')
        pos_bases = {}  # {pos: ['A', 'T', 'T', 'A', ..]},  # order by alleles
        nhaps = 0

        for rec in chr_reader:
            if start and rec.POS < start:
                continue
            if end and rec.POS > end:
                logging.info('exceeds end position %s: pos:%s, ref:%s, alt:%s', end, rec.POS, rec.REF, rec.ALT)
                break
            if args.snp_only and (len(str(rec.REF)) != 1 or any(len(str(alt)) != 1 for alt in rec.ALT)):
                logging.info('skipped pos:%s, ref:%s, alt:%s', rec.POS, rec.REF, rec.ALT)
                continue

            ref_alt_bases = [str(rec.REF)] + map(str, rec.ALT)

            def gt2bases(gt):
                nums = splitter.split(gt)
                return [ref_alt_bases[int(n)] for n in nums]

            #bases = np.array([gt2bases(gt['GT']) for gt in rec.samples])
            #logging.info(bases)
            bases = np.array([gt2bases(gt['GT']) for gt in rec.samples]).flatten()
            nhaps = len(bases)
            major_count = max(len(tuple(b for b in bases if b == base)) for base in set(bases))
            maf = 1. * (nhaps - major_count) / nhaps
            if maf < args.min_maf:
                logging.info('skipped pos:%s, ref:%s, alt:%s (maf: %s < %s)', rec.POS, rec.REF, rec.ALT, maf, args.min_maf)
                continue

            logging.info('add pos:%s, ref:%s, alt:%s', rec.POS, rec.REF, rec.ALT)
            pos_bases[rec.POS] = bases

        poss = sorted(pos_bases)
        pos_str = ','.join(map(str, poss))
        print ('#chrom={0}'.format(chrom1), sep='\t')
        print ('#pos={0}'.format(pos_str), sep='\t')
        print ('#npos={0}'.format(len(poss)), sep='\t')

        if nhaps:
            bases_list = [','.join(pos_bases[pos][i] for pos in poss) for i in xrange(nhaps)]
            if args.with_weight:
                for (i, bases) in enumerate(set(bases_list)):  # unique bases
                    weight = len(tuple(1 for bases1 in bases_list if bases1 == bases))
                    print (i + 1, bases, weight, sep='\t')
            else:
                for (i, bases) in enumerate(bases_list):
                    print (i + 1, bases, sep='\t')

        break


def read_haplotypes(lines):
    """
    Input:
        TSV file with following columns

        1. haplotype id
        2. bases1 (e.g. A, T, G, C, AT, AGG, ...)
        3. bases2
        ...

    """
    haps = {}
    for line in lines:
        dat = line.rstrip('\n').split('\t')
        haps[dat[0]] = dat[1:]
    return haps


@command.add_sub
@argument('haplotypes', help='aligned sequence')
@argument('-p', '--ploidies', default='2:4,3:4,4:4', help='population of each ploidy [%(default)s]')
def gen_alleles(args):
    """
    Input:
        see function read_haplotypes

    Outut:
        TSV file with following columns

        1. sample id
        2. haplotype id
        3. bases1
        4. bases2 ...
    """
    ploidies = list((int(ploidy), int(n)) for (ploidy, n)
                      in (x.split(':') for x in args.ploidies.split(',')))
    haps = read_haplotypes(open(args.haplotypes))
    hap_ids = haps.keys()
    sample_id = 0

    for (p, n) in ploidies:
        for _ in xrange(n):
            sample_id += 1
            for _ in xrange(p):
                hap_id = np.random.choice(hap_ids)
                print (sample_id, hap_id, *haps[hap_id], sep='\t')


def read_haplotypes2(lines):
    """
    Input:
        TSV file with following columns
        1. hap id
        2. comma separated bases
        3. nsamples

    Output:
        {hap_id: {'bases': [base], 'nsamples': number of samples}}
    """
    haps = {}
    for line in lines:
        if line.startswith('#'):
            continue
        (hap_id, bases, nsamples) = line.rstrip('\n').split('\t')
        haps[hap_id] = {'bases': bases.split(','), 'nsamples': int(nsamples)}

    return haps


@command.add_sub
@argument('haplotypes', help='aligned sequence')
@argument('-p', '--ploidies', default='2:4,3:4,4:4', help='population of each ploidy [%(default)s]')
def gen_alleles2(args):
    """ Emit random choosed alleles without replacement

    Input:
        1. hap id
        2. comma separated bases
        3. nsamples

    Outut:
        TSV file with following columns

        1. sample id
        2. haplotype id
        3. bases1
        4. bases2 ...
    """
    ploidies = list((int(ploidy), int(n)) for (ploidy, n)
                      in (x.split(':') for x in args.ploidies.split(',')))
    haps = read_haplotypes2(open(args.haplotypes))

    hap_ids = []
    for hap_id in haps:
        hap_ids.extend([hap_id] * haps[hap_id]['nsamples'])

    np.random.shuffle(hap_ids)  # overwrite hap_ids !

    sample_id = 0
    for (p, n) in ploidies:
        for _ in xrange(n):
            sample_id += 1
            for _ in xrange(p):
                hap_id = hap_ids.pop()
                print (sample_id, hap_id, *haps[hap_id]['bases'], sep='\t')


def init_hist(bases=list('ATGC')):
    return dict((b, 0) for b in bases)


def sample_hist(base, mean_depth, error=0.01, bases=list('ATCG')):
    depth = np.random.poisson(mean_depth)
    e_depth = np.random.binomial(depth, error) if depth > 0 else 0
    hist = init_hist(bases=bases)
    hist[base] += depth - e_depth
    for e_base in np.random.choice(filter(lambda b: b != base, bases), e_depth):
        hist[e_base] += 1
    return hist


def iter_seq(lines):
    for line in lines:
        row = line.rstrip('\r\n').split('\t')
        sample_id = row[0]
        hap_id = row[1]
        seq = row[2:]
        yield (sample_id, hap_id, seq)


@command.add_sub
@argument('alleles', help='see gen_alleles command')
@argument('--depth', type=int, default=15, help='mean depth per haploid')
@argument('--error', type=float, default=0.01, help='mean error rate per base')
def gen_hist(args):
    """
    Input:
        output of gen_alleles

    Output:

    1. sample id
    2. base (A|T|C|G)
    3. count at pos 1
    4. count at pos 2
    ...
    """
    bases = 'ATCG'

    for (sample_id, grouped) in groupby(iter_seq(open(args.alleles)), lambda x:x[0]):
        histss = []
        for (_, hap_id, seq) in grouped:
            histss.append([sample_hist(base, args.depth, error=args.error) for base in seq])

        hists = [init_hist() for _ in histss[0]]
        for hists1 in histss:
            for (h, h1) in zip(hists, hists1):
                for (b, v) in h1.items():
                    h[b] += v

        for b in bases:
            print (sample_id, b, *(h[b] for h in hists), sep='\t')


@command.add_sub
@argument('-r', '--result-file', required=True, help='Result file')
@argument('-a', '--allele-file', help='Allele file')
@argument('-d', '--mean-depth', type=float, help='mean depth of haploid')
def evaluate_alleles(args):
    """
    Input:
        json
    """
    def parse_lda_result(result_file):
        result = json.loads(open(result_file).read())
        hap_bases = np.array(result['haps']).transpose()
        hap_probs = np.array(result['hap_probs']).transpose()
        gammas = np.array(result['gamma'])
        pred_w1 = (gammas / gammas.sum(axis=1)[:,np.newaxis]).mean(axis=0)  # [[hap ratio of K_n haps] of N samples] => [mean hap ratio of K haps]
        logging.info('pred_w1 : %s', pred_w1)
        logging.info('sum of hap_probs : %s', (1 - hap_probs).sum(axis=1).flatten())
        summary = {}
        summary['K'] = len(hap_bases)
        summary['nsites'] = len(hap_bases[0])
        for bases in hap_bases:
            logging.info(''.join(bases))
        summary['ll'] = result['log_likelihood']
        summary['hap_uncertainty'] = (1 - hap_probs).sum(axis=1).mean()
        summary['hap_uncertainty_w1'] = (pred_w1 * (1 - hap_probs).sum(axis=1).flatten()).sum()
        return summary

    def parse_lda_result_allele(result_file, allele_file):
        result = json.loads(open(result_file).read())
        hap_bases = np.array(result['haps']).transpose()
        hap_probs = np.array(result['hap_probs']).transpose()

        true_haps = {}
        gammas = np.array(result['gamma'])
        pred_w1 = (gammas / gammas.sum(axis=1)[:,np.newaxis]).mean(axis=0)  # [[hap ratio of K_n haps] of N samples] => [mean hap ratio of K haps]
        logging.info('pred_w1 : %s', pred_w1)
        # pred_w2 # TODO
        # true_w2 # TODO
        true_sample_haps = {}  # {sample_id: {hap_id: count}}
        true_sample_base_weights = {}   # {sample_id: [{base: weights}]}
        true_sample_ploidies = defaultdict(int)  # {sample_id: ploidy}
        seq_len = 0
        hap_counts = defaultdict(int)  # {hap_id: count}
        sample_ids = []

        for (sample_id, grouped) in groupby(iter_seq(open(args.allele_file)), lambda x:x[0]):
            #sample_id = int(sample_id)
            sample_ids.append(sample_id)
            true_sample_haps[sample_id] = defaultdict(int)

            for (_, hap_id, seq) in grouped:
                hap_id = int(hap_id)
                true_haps[hap_id] = seq
                true_sample_haps[sample_id][hap_id] += 1
                seq_len = len(seq)
                hap_counts[hap_id] += 1
                true_sample_ploidies[sample_id] += 1

            ploidy = true_sample_ploidies[sample_id]
            true_sample_base_weights[sample_id] = [defaultdict(float) for _ in xrange(seq_len)]
            for (hap_id, count) in true_sample_haps[sample_id].items():
                seq = true_haps[hap_id]
                for (i, base) in enumerate(seq):
                    true_sample_base_weights[sample_id][i][base] += 1. * count / ploidy

        # [[hap count of K_n haps] of N samples]
        # logging.info(sample_ids)
        # logging.info(true_sample_haps)
        # logging.info(true_haps)
        # logging.info(hap_bases)
        sample_hap_counts = np.array([[true_sample_haps[sample_id][hap_id] for sample_id in sample_ids]
                                      for hap_id in sorted(true_haps)]).transpose()

        true_base_weights = {}  # {hap_id: [weight of each base]}
        base_hap_counts_list = [defaultdict(int) for i in xrange(seq_len)]  # [{base: count} for M poss]

        for hap_id in true_haps:
            seq = true_haps[hap_id]
            for (i, base) in enumerate(seq):
                base_hap_counts_list[i][base] += hap_counts[hap_id]

        for hap_id in true_haps:
            seq = true_haps[hap_id]
            base_scores = - np.log(np.array([1. * counts[base] / sum(counts.values()) for (base, counts) in zip(seq, base_hap_counts_list)]))
            true_base_weights[hap_id] = base_scores / base_scores.sum()  # weight of each base of each haplotype

        #logging.warning(sample_hap_counts)

        pred_max_rates = {}     # bases are equally weighted
        true_max_rates = {}
        bw_pred_max_rates = {}  # bases are weighted by - log (base allele freq)
        bw_true_max_rates = {}
        sample_gt_deviances = {}   # {sample_id: concordance}

        for (i, i_gammas) in enumerate(gammas):
            sample_id = sample_ids[i]
            pred_hap_weights = i_gammas / i_gammas.sum()
            pred_base_weights = [defaultdict(float) for _ in xrange(seq_len)]

            for (ratio, bases) in zip(pred_hap_weights, hap_bases):
                for (j, base) in enumerate(bases):
                    pred_base_weights[j][base] += ratio

            #pred_base_ratio = pred_hap_ratio * 
            alphabets = result['alphabets']
            site_devs = np.array([np.abs([true_bw[b] - pred_bw[b] for b in alphabets]).sum() for (true_bw, pred_bw) in zip(true_sample_base_weights[sample_id], pred_base_weights)])
            sample_gt_deviances[sample_id] = site_devs.mean()
            #logging.warning(site_devs)


        for (pred_id, bases) in enumerate(hap_bases):
            for (true_id, true_seq) in true_haps.items():
                match_rate = 1. * len([_ for (base, true_base) in zip(bases, true_seq) if base == true_base]) / len(true_seq)
                #logging.warning(bw_match_rate)
                pred_max_rates[pred_id] = max(match_rate, pred_max_rates.get(pred_id, 0))
                true_max_rates[true_id] = max(match_rate, true_max_rates.get(true_id, 0))

        pred_max_rate_arr = np.array([pred_max_rates[hap_id] for hap_id in sorted(pred_max_rates)])
        true_max_rate_arr = np.array([true_max_rates[hap_id] for hap_id in sorted(true_max_rates)])

        summary = {}
        summary['true_K'] = len(true_haps)
        summary['K'] = len(hap_bases)
        summary['nsites'] = len(hap_bases[0])
        summary['ll'] = result['log_likelihood']
        summary['hap_uncertainty'] = (1. * (1 - hap_probs).sum(axis=1)).mean()

        summary['precision'] = p = pred_max_rate_arr.mean()
        summary['recall'] = r = true_max_rate_arr.mean()
        summary['f_measure'] = p * r / ((p + r) / 2.)

        summary['sample_gt_deviance'] = np.mean(sample_gt_deviances.values())

        return summary

    if args.allele_file:
        summary = parse_lda_result_allele(args.result_file, args.allele_file)
        labels = ['true_K', 'K', 'nsites', 'll',
                  'hap_uncertainty', 'precision', 'recall', 'f_measure', 'sample_gt_deviance',
                  ]
    else:
        summary = parse_lda_result(args.result_file)
        labels = ['K', 'nsites', 'll', 'hap_uncertainty',
                ]

    print (*labels, sep='\t')
    print (*[summary[label] for label in labels], sep='\t')
    # log likelihood lower bound

    # haplotype concordance
    #print ('haplotype precision:', result['precision'], sep='\t')
    #print ('haplotype recall:', result['recall'], sep='\t')

    # individual genotype concordance
    #print ('genotype concordance:', result['genotype_concordance'], sep='\t')


@command.add_sub
@argument('histfile')
@argument('-K', type=int, default=3, help='number of haplotypes')
@argument('-i', '--init-haplotypes', help='comma separated list of initial haplotypes ([ATCG]+[,$])+')
@argument('--ll-diff', type=float, default=1e-4, help='Convergence criterion for the log liklihood [%(default)s]')
@argument('--use-ll-offset', default=False, action='store_true', help='Consider symmetric solutions in evaluation of the log likelihood')
def estimate_alleles(args):
    """
    """
    if args.init_haplotypes:
        haps = [list(hap) for hap in args.init_haplotypes.split(',')]
    else:
        haps = None

    lda_data = estimator.LDAData(args.histfile)
    lda_status = estimator.LDAStatus.init_with_data(lda_data, nhaps=args.K, haplotypes=haps, use_ll_offset=args.use_ll_offset)
    ll_prev = None
    ll_diff = args.ll_diff
    while ll_prev is None or abs(ll_prev - lda_status.ll) > ll_diff:
        logging.info(lda_status)
        ll_prev = lda_status.ll
        lda_status = lda_status.get_next()

    print (json.dumps(lda_status.as_dict()))


_attrs = tuple('indel is_del qpos mq base qual start_clipped end_clipped clipped skip_flag'.split())

class BAM_FLAGS:
    unmapped = 0x004
    secondary = 0x100
    supplementary = 0x800

SKIP_FLAG = BAM_FLAGS.unmapped | BAM_FLAGS.secondary | BAM_FLAGS.supplementary # unmapped, secondary or supplementary

class ReadInfo(object):
    __slots__ = _attrs

    def __init__(self, rec, skip_flag=SKIP_FLAG):
        self.indel = rec.indel
        self.is_del = rec.is_del
        #rec.is_head
        #rec.is_tail
        #self.level = rec.level
        self.qpos = rec.query_position
        a = rec.alignment
        self.base = a.seq[self.qpos]
        self.qual = a.qual[self.qpos]
        #print (dir(a))
        self.start_clipped = (a.qstart > 0)
        self.end_clipped = (a.qend < a.rlen)
        self.clipped = self.start_clipped or self.end_clipped
        self.mq = a.mapq
        self.skip_flag = a.flag & skip_flag

    def __str__(self):
        return ' '.join(['{0}={1}'.format(a, getattr(self, a)) for a in _attrs])


class ReadHist(object):
    __slots__ = ('N A T C G D ndel nins'.split())

    def __init__(self):
        for a in self.__slots__:
            setattr(self, a, 0)

    def __str__(self):
        return '\t'.join(['{0}:{1}'.format(a, getattr(self, a)) for a in self.__slots__])

    def is_segregate(self):
        return len(filter(lambda x: x > 0, (self.A, self.T, self.C, self.G, self.D)) > 1) or self.nins > 0 or self.ndel > 0

    def major_base(self):
        return ('A', 'T', 'C', 'G', 'D')[np.array((self.A, self.T, self.C, self.G, self.D)).argmax()]

    def total_count(self):
        return self.A + self.T + self.C + self.G + self.D

    def major_freq(self):
        try:
            return 1. * getattr(self, self.major_base()) / self.total_count()
        except ZeroDivisionError:
            return float('nan')

    def add_read(self, rinfo):
        if rinfo.indel > 0:
            self.nins += 1
        elif rinfo.indel < 0:
            self.ndel += 1
        #else:
        if rinfo.is_del:  # deletion base
            self.D += 1
        else:
            setattr(self, rinfo.base, getattr(self, rinfo.base) + 1)


@command.add_sub
@argument('bam', help='')
@argument('-r', '--region')
@argument('-M', '--max-major-freq', type=float, default=1)
@argument('-g', '--genome', help='reference fasta file (with .fai file)')
@argument('-H', '--no-header', dest='with_header', default=True, action='store_false', help='no header')
@argument('--use-multimaps', default=False, action='store_true')
def bam2hist(args):
    samfile = pysam.Samfile(args.bam)

    regions = samfile
    (rname, start, end) = parse_region(args.region)

    refseq = None
    if args.genome:
        fasta = pysam.Fastafile(args.genome)
        refseq = fasta.fetch(reference=rname, start=start, end=end)

    if args.with_header:
        print ('rname', 'pos', 'offset',
               'ref',
               'nread', 'major_base', 'major_count', 'minor_count',
               'major_freq', 'minor_freq',
               'N', 'A', 'T', 'C' ,'G', 'D', 'ndel', 'nins',
              sep='\t')

    if args.use_multimaps:
        skip_flag = BAM_FLAGS.unmapped
    else:
        skip_flag = SKIP_FLAG

    for p in samfile.pileup(reference=rname, start=start, end=end, mask=skip_flag):
        if (p.pos < start) or (end is not None and end <= p.pos):  # skip outside of region
            continue

        h = ReadHist()
        for read in p.pileups:
            if read.query_position is None:
                continue
            info = ReadInfo(read, skip_flag=skip_flag)
            if info.skip_flag:
                continue
            h.add_read(info)
        major_base = h.major_base()
        major_count = getattr(h, major_base)
        minor_count = h.total_count() - major_count
        major_freq = h.major_freq()
        if major_freq > args.max_major_freq:
            continue
        print (rname, p.pos + 1, p.pos - start,
               '.' if refseq is None else refseq[p.pos - start],
               p.n, major_base, major_count, minor_count,
               '{0:.3f}'.format(major_freq), '{0:.3f}'.format(1 - major_freq),
               h.N, h.A, h.T, h.C, h.G, h.D, h.ndel, h.nins,
               sep='\t')


if __name__ == '__main__':
    command.run()
