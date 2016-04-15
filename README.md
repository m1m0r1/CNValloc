CNValloc
======================

<img alt="CNValloc" src="https://raw.githubusercontent.com/m1m0r1/CNValloc/master/figures/cnvalloc_logo.png" width="50%" />

Requirements
----------------------
- Python 2.6 or 2.7
- Modules: numpy, scipy, argtools


Description
----------------------
A tool for estimating sequences of CNV alleles from multiple individuals.
The allele ratio of each sample is also inferred.


### A simple example

```sh
python cnvalloc estimate_alleles -v -K 4 examples/hist.txt
```

The result of estimation is emitted as JSON format.


Check performance for the number of haplotypes K = 1..10

```sh
$ parallel -k 'python cnvalloc estimate_alleles -K {} examples/hist.txt | python cnvalloc evaluate_alleles -r /dev/stdin -a examples/haps.txt' ::: {1..10}
```

* examples/haps.txt : True hpalotypes for evaluation
    - column 1: Sample id
    - column 2: Allele id
    - column n (n>2): The base of the allele at the n-2 th variant site

* examples/hist.txt : Histogram of data
    - column 1: Sample id
    - column 2: One of the 'ATCG' bases
    - column n(n>2) : The number of observed bases at n-2 th variant site



### Workflow for BAM files

1. Make pileup histograms from BAM files

```sh
$ python cnvalloc bam2hist {BAM file n} -r chr1:10000000-100010000 > pileups.n.txt
```

2. Import the pileup files to a database such as sqlite3
3. Select the variable sites to use with some criteria (e.g. minor_count > 15 for any of the samples)
4. Create an input file for the `cnvalloc estimate_alleles` by querying to the database


Future work
----------------------
- Consider variant types other than mutations
- Write tools for step 2-4 of the above workflow


References
----------------------
T. Mimori et al, 2015 BMC Bioinformatics
__"Estimating copy numbers of alleles from population-scale high-throughput sequencing data"__

http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-16-S1-S4
