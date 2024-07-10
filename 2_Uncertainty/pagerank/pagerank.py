import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus: dict[str, set[str]], page: str, damping_factor: float):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    if len(links) == 0:
        links_probability = 0
    else:
        links_probability = damping_factor / len(links)
    all_probability = (1 - damping_factor) / len(corpus)
    return {
        key: links_probability + all_probability if key in links else all_probability
        for key in corpus
    }


def sample_pagerank(corpus: dict[str, set[str]], damping_factor: float, n: int):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    def add_sample(page):
        if page not in samples:
            samples[page] = 1
        else:
            samples[page] += 1

    # samples is a dict having as key the page (e.g. 1.html)
    # and as a value the amount of times this page was chosen
    samples = {}

    # Choose page at random and add to samples
    pages = list(corpus.keys())
    page = random.choice(pages)
    add_sample(page)

    # Loop through every sample chosing next page based on transitional model
    for _ in range(n - 1):
        distribution = transition_model(corpus, page, damping_factor)
        page = random.choices(list(distribution.keys()), 
                              weights=list(distribution.values()), k=1)[0]
        add_sample(page)

    # Normalize samples to sum up to 1
    total_sum = sum(samples.values())
    normalized_samples = {key: value / total_sum for key, value in samples.items()}
    
    return normalized_samples


def iterate_pagerank(corpus: dict[str, set[str]], damping_factor: float):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Interprete a page that has no links as page that that
    # has links to every page including itself
    corpus = corpus.copy()
    for key, value in corpus.items():
        if len(value) == 0:
            for key in corpus:
                value.add(key)
    rev_corpus = links_to_pages(corpus)

    starting_probability = 1 / len(corpus)
    distribution = {key: starting_probability for key in corpus}

    factor1 = (1 - damping_factor) / len(corpus)

    # Adjust the propability distribution until no PageRank value changes by
    # more that 0.0001 between the current rank values and the new rank values
    stop = False
    while not stop:
        stop = True
        for key, value in distribution.items():
            pages = rev_corpus[key]
            total_sum = 0
            for page in pages:
                total_sum += distribution[page] / len(corpus[page])
            factor2 = damping_factor * total_sum
            new_value = factor1 + factor2
            distribution[key] = new_value
            if abs(value - new_value) > 0.001:
                stop = False

    return distribution


def links_to_pages(corpus: dict[str, set[str]]):
    """
    Return a dict with key, value pairs where key is the page (e.g. 1.html) and value is a set containing pages that link to the page in the key
    """
    rev_corpus = {}
    for page in corpus:
        links = set()
        for key, value in corpus.items():
            if page in value:
                links.add(key)
        rev_corpus[page] = links
    return rev_corpus


if __name__ == "__main__":
    main()
