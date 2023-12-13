import torch
file_path = "<PATH_TO_THIS_FOLDER>"
eplus_location = "<PATH_TO_ENERGYPLUS>"


def str_to_bool(s: str) -> bool:
    """
    The str_to_bool function takes a string as input and returns True if the string is 'True', 'true', or 1.
    Otherwise, it returns False.

    :param s: Convert the string to a boolean
    :return: A boolean value
    """
    if s.lower() == "true" or s == '1' or s.lower() == 't':
        return True
    return False


def to_numpy(tensor):
    """
    The to_numpy function takes a PyTorch tensor and returns the equivalent numpy array.

    :param tensor: Convert the tensor to a numpy array
    :return: The tensor as a numpy array
    """
    return tensor.to("cpu").detach().numpy()


def to_torch(arr):
    """
    The to_torch function takes a numpy array and converts it to a torch tensor.
    It also moves the tensor to the GPU if one is available, and sets requires_grad=True so that we can compute gradients with respect to this variable later.

    :param arr: Pass in the array that is to be converted into a tensor
    :return: A tensor from an array
    """
    return torch.tensor(arr, device="cuda", requires_grad=True)