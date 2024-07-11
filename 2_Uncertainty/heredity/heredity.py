import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    def get_parents_heredity_prob(person: str) -> tuple[float, float, float, float]:
        """
        Calculate the probability of getting gene from mother or father based on the rules of heredity and the mutation
        """
        mother = people[person]["mother"]
        father = people[person]["father"]
        mother_genes = (
            2 if mother in two_genes else 1 if mother in one_gene else 0
        )
        father_genes = (
            2 if father in two_genes else 1 if father in one_gene else 0
        )

        mother_prob = 1 - PROBS["mutation"] if mother_genes == 2 \
            else 0.5 if mother_genes == 1 else PROBS["mutation"]
        not_mother_prob = PROBS["mutation"] if mother_genes == 2 \
            else 0.5 if mother_genes == 1 else 1 - PROBS["mutation"]
        father_prob = 1 - PROBS["mutation"] if father_genes == 2 \
            else 0.5 if father_genes == 1 else PROBS["mutation"]
        not_father_prob = PROBS["mutation"] if father_genes == 2 \
            else 0.5 if father_genes == 1 else 1 - PROBS["mutation"]

        return mother_prob, not_mother_prob, father_prob, not_father_prob
    
    probs = []

    for person in people:

        # Person with 1 gene
        if person in one_gene:
            # Calculate probability for having 1 gene based on having parents information or not
            if not people[person]["father"]:
                gene_prob = PROBS["gene"][1]
            else:
                # 2 possibilities - gene from mother or from father
                mother_prob, not_mother_prob, father_prob, not_father_prob = (
                    get_parents_heredity_prob(person)
                )
                gene_prob = (mother_prob * not_father_prob) + \
                    (father_prob * not_mother_prob)

            if person in have_trait:
                trait_prob = PROBS["trait"][1][True]
            else:
                trait_prob = PROBS["trait"][1][False]

        # Person with 2 genes
        elif person in two_genes:
            # Calculate probability for having 1 gene based on having parents information or not
            if not people[person]["father"]:
                gene_prob = PROBS["gene"][2]
            else:
                # 1 possiblity - 1 gene from mother and 1 gene from father
                mother_prob, not_mother_prob, father_prob, not_father_prob = (
                    get_parents_heredity_prob(person)
                )
                gene_prob = mother_prob * father_prob

            if person in have_trait:
                trait_prob = PROBS["trait"][2][True]
            else:
                trait_prob = PROBS["trait"][2][False]

        # Person with 0 genes
        else:
            # Calculate probability for having 0 genes based on having parents information or not
            if not people[person]["father"]:
                gene_prob = PROBS["gene"][0]
            else:
                # 1 possibility - none of parent give a gene
                mother_prob, not_mother_prob, father_prob, not_father_prob = (
                    get_parents_heredity_prob(person)
                )
                gene_prob = not_mother_prob * not_father_prob

            if person in have_trait:
                trait_prob = PROBS["trait"][0][True]
            else:
                trait_prob = PROBS["trait"][0][False]
            
        # Combine probability for a person knowing the probability of having
        # specific amount of genes and probability of having the trait
        probs.append(gene_prob * trait_prob)
                
    # Calculate joint probability by multiplying values for each of the people
    joint_probability = 1
    for prob in probs:
        joint_probability *= prob

    return joint_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        total_g = (
            probabilities[person]["gene"][0]
            + probabilities[person]["gene"][1]
            + probabilities[person]["gene"][2]
        )
        scaling_factor = 1 / total_g
        probabilities[person]["gene"][0] *= scaling_factor
        probabilities[person]["gene"][1] *= scaling_factor
        probabilities[person]["gene"][2] *= scaling_factor

        total_t = (
            probabilities[person]["trait"][True]
            + probabilities[person]["trait"][False]
        )
        scaling_factor = 1 / total_t
        probabilities[person]["trait"][True] *= scaling_factor
        probabilities[person]["trait"][False] *= scaling_factor


if __name__ == "__main__":
    main()
