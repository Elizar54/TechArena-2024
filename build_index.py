import numpy as np
from kmeans import My_KMeans
import sys
import pickle
import struct


def read_fvecs(fp):
    '''
    Reads a binary file containing floating-point vectors in the fvecs format.

    This function reads a binary file where the first integer indicates the dimensionality of 
    the vectors, followed by the vectors themselves. Each vector is represented as a sequence 
    of floating-point numbers. The function returns these vectors as a NumPy array of type 
    float32.

    Parameters:
    ----------
    fp : file-like object or str
        A file pointer or the path to the binary file to read. The file should be in the fvecs 
        format, which starts with an integer indicating the dimension of the vectors.

    Returns:
    -------
    np.ndarray
        A NumPy array of shape (n, dim) where n is the number of vectors and dim is the 
        dimensionality of each vector. The array contains the vector data as float32.
    '''
    array = np.fromfile(fp, dtype='int32')
    dim = array[0]
    return array.reshape(-1, dim + 1)[:, 1:].copy().view('float32')


def code_to_uint8(dataset):
    '''
    Codes the data given from float32 format to uint8 format.

    Parameters:
    ------------
    data_batch : numpy array
        Data that has to be coded.
    
    Returns:
    ----------
    numpy.ndarray
        A Coded array.
    '''
    return (dataset * 255).astype(np.uint8)


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

    with open(file, 'rb') as file:
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


def remove_brackets(query):
    return query.replace('[', '').replace(']', '')


def build_index(filename):
    '''
    Builds an index for a dataset by applying KMeans clustering and saving the results.

    The function checks the filename to determine the dataset type (SIFT or GIST) and 
    performs clustering accordingly. For SIFT datasets, it uses a fixed number of clusters 
    and saves the clustered dataset along with the KMeans model. For GIST datasets, it 
    processes the data in batches, applies KMeans clustering, and saves the results in a 
    specific binary format.

    Parameters:
    ----------
    filename : str
        The path to the dataset file. The function distinguishes between SIFT and GIST datasets 
        based on the presence of 'sift' in the filename.

    Returns:
    -------
    None
        The function does not return any value but saves the clustered dataset and KMeans model 
        to disk.
    '''

    if 'sift' in filename:
        clusters = 4
        dataset = read_fvecs(filename)
        kmeans = My_KMeans(n_clusters=clusters)
        labels = kmeans.fit(dataset[:100000])
        labels = kmeans.predict(dataset)
        dataset = np.column_stack([dataset, labels])

        np.save('matrix_sift.npy', dataset)

        with open('kmeans_model_sift.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
    else:
        clusters = 5

        dataset = fvecs_read_batch(file=filename, dim=960, start=0, batch_size=100000)

        kmeans = My_KMeans(n_clusters=clusters)
        labels = kmeans.fit(dataset)
        dataset = np.column_stack([dataset, labels])
        dataset = code_to_uint8(dataset=dataset)

        file_path = 'vectors_gist.uint8'

        with open(file_path, 'wb') as f:
            num_vectors = 1_000_000
            f.write(np.array(num_vectors, dtype=np.int32).tobytes())
            
            vector_dim = dataset.shape[1]
            f.write(np.array(vector_dim, dtype=np.int32).tobytes())

            f.write(dataset.tobytes())

        
        for start in range(100_000, 900_001, 100_000):
            dataset = fvecs_read_batch(file=filename, dim=960, start=start, batch_size=100_000)
            labels = kmeans.predict(dataset)
            dataset = np.column_stack([dataset, labels])
            dataset = code_to_uint8(dataset=dataset)

            file_path = 'vectors_gist.uint8'

            with open(file_path, 'ab') as f:
                
                f.write(dataset.tobytes())
 
        with open('kmeans_model_gist.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        

if __name__ == "__main__":
    filename = sys.argv[1]
    build_index(filename)
