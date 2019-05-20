import csv

def get_brain_body_data(csv_file):
    """
    Load brain - weight to test linear regression

    The data records the average weight of the brain and body for a number of mammal species
    More details here: http://people.sc.fsu.edu/~jburkardt/datasets/regression/x01.txt

    Parameters
    :param csv_file: basestring
        Path of csv file containing data

    Returns
    :return: lists
        List of body and brain weight
    """
    body_weight = []
    brain_weight = []

    with open(csv_file, 'rt') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            if row:
                idx, brain_w, body_w = row[0].split()
                brain_weight.append(float(brain_w))
                body_weight.append(float(body_w))

    return body_weight, brain_weight