import pickle
import statistics

# file_path is a .pickle file
def save_data(data, file_path: str):
    file = open(file_path, 'wb')
    pickle.dump(data, file)
    file.close()
    return True

def read_data(file_path: str):
    with open(file_path, 'rb') as file:
        data =pickle.load(file)
        return data
    

def get_mean(data: list) -> float:
    return statistics.mean(data)

def get_standard_deviation(data: list) -> float:
    return statistics.stdev(data)
