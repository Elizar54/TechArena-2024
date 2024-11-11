import numpy as np
import struct
import sys
import pickle


def fvecs_read_batch(file, dim, start, batch_size):
    '''
    Reads a batch of floating-point vectors from a binary file.

    This function reads a specified number of vectors (batch size) from a binary file containing 
    floating-point vectors. Each vector is preceded by an integer that specifies its dimensionality. 
    The function seeks to the appropriate position in the file based on the provided start index and 
    reads the vectors into a list.

    Parameters:
    ----------
    file : file-like object
        A file object that has been opened in binary mode. It should point to a file formatted with 
        floating-point vectors.

    dim : int
        The dimensionality of each vector. This value is used to validate the size of each vector read 
        from the file.

    start : int
        The starting index (zero-based) from which to read the batch of vectors. Each vector occupies 
        (dim + 1) * 4 bytes in the file.

    batch_size : int
        The number of vectors to read in this batch.

    Returns:
    -------
    np.ndarray
        A numpy array of shape (batch_size, dim) containing the floating-point vectors read from the file.
    '''
    
    file.seek(start * (dim + 1) * 4)

    buffer = file.read(batch_size * (dim + 1) * 4)
    if not buffer:
        raise IOError("Error reading batch from file.")

    vectors = []
    for i in range(batch_size):
        vector_dim = struct.unpack('i', buffer[i * (dim + 1) * 4: i * (dim + 1) * 4 + 4])[0]
        if vector_dim != dim:
            raise ValueError("Non-uniform vector sizes in file.")
        vector = struct.unpack('f' * dim, buffer[i * (dim + 1) * 4 + 4: (i + 1) * (dim + 1) * 4])
        vectors.append(vector)

    return np.array(vectors)


def euclidean_distance(vec1, vec2):
    '''
    Calculates the euclidean distance between two vectors.

    Parameters:
    ------------
    vec1 : numpy array
    vec2 : numpy array

    Returns:
    ---------
    float
        Euclidean distance between given vectors.
    '''
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def decode_data(data_batch):
    '''
    Decodes the data given from uint8 format to float32.

    Parameters:
    ------------
    data_batch : numpy array
        Data that has to be decoded.
    
    Returns:
    ----------
    numpy.ndarray
        A decoded array.
    '''
    return data_batch.astype(np.float32) / 255.0


def read_batches_uint8(file_path, batch_size):
    '''
    Reads batches of uint8 vectors from a binary file.

    This generator function reads a specified number of vectors (batch size) from a binary file containing 
    uint8 vectors. The first two 4-byte integers in the file represent the total number of vectors and 
    the dimensionality of each vector, respectively. The function yields batches of vectors until all 
    vectors have been read.

    Parameters:
    ----------
    file_path : str
        The path to the binary file containing the vectors.
    
    batch_size : int
        The number of vectors to read in each batch. Must not exceed the total number of vectors in the file.

    Yields:
    -------
    numpy.ndarray
        A numpy array of shape (current_batch_size, vector_dim) containing the uint8 vectors for the current batch.
    '''

    with open(file_path, 'rb') as f:
        num_vectors = np.frombuffer(f.read(4), dtype=np.int32)[0]

        vector_dim = np.frombuffer(f.read(4), dtype=np.int32)[0]

        if batch_size > num_vectors:
            raise ValueError("Batch size cannot be larger than the number of vectors.")

        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            current_batch_size = end - start
            
            bytes_to_read = current_batch_size * vector_dim * np.dtype(np.uint8).itemsize
            
            data_batch = np.frombuffer(f.read(bytes_to_read), dtype=np.uint8).reshape(current_batch_size, vector_dim)
            
            yield data_batch
    

def remove_brackets(query):
    return query.replace('[', '').replace(']', '')


def code_to_uint8(data):
    '''
    Converts floating-point data to unsigned 8-bit integers.

    This function takes an array of floating-point numbers and 
    scales them to the range [0, 255]. The resulting values are then converted to unsigned 
    8-bit integers (uint8). 

    Parameters:
    ----------
    data : np.ndarray
        A numpy array of floating-point numbers. 

    Returns:
    -------
    np.ndarray
        A numpy array of the same shape as data, containing the scaled values as unsigned 
        8-bit integers (uint8)
    '''
    return (data * 255).astype(np.uint8)


def main(query):
    '''
    The main function that search 10 approximate nearest neighbours.

    Parameters:
    ------------
    query: str
        A string that consists the information about the vector whose 10 nearest neighbours are to be found.

    Prints:
    --------
    Indices of 10 approximate nearest neighbours of given vector. 
    '''

    query = remove_brackets(query)

    if len(query) > 3000:
        with open('kmeans_model_gist.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        
        query_vector = np.array([float(x) for x in query.split(',')]).reshape(1, -1)
        cluster = kmeans.predict(query_vector)[0]
        query_vector = code_to_uint8(np.append(query_vector, cluster))

        distances = []
        batch_size = 10000
        c = 0

        for batch_vectors in read_batches_uint8('vectors_gist.uint8', batch_size):
            for i in range(batch_size):
                if batch_vectors[i][-1] == query_vector[-1]:
                    vec1 = decode_data(batch_vectors[i][:-1]) 
                    vec2 = decode_data(query_vector[:-1]) 
                    dist = euclidean_distance(vec1, vec2)
                    distances.append((dist, c + i))
            c += batch_size

    else:
        with open('kmeans_model_sift.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        dataset = np.load('matrix_sift.npy')

        query_vector = np.array([float(x) for x in query.split(',')]).reshape(1, -1)

        cluster = kmeans.predict(query_vector)[0]

        search_data = dataset[:, len(query_vector[0])] == cluster
        num_vectors = len(dataset)

        dataset = dataset[:, :len(query_vector[0])]

        distances = []
        for i in range(num_vectors):
            if search_data[i]:
                dist = euclidean_distance(query_vector, dataset[i])
                distances.append((dist, i))

    distances.sort(key=lambda x: x[0])

    result = ','.join(str(idx) for _, idx in distances[:10])
    print(result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <query vector>")
        sys.exit(1)

    query = sys.argv[1]
    main(query)
