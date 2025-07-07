import torch

def convolveList(x, w, s, p, d, g):
    return torch.nn.functional.conv2d(torch.tensor(x).unsqueeze(0), torch.tensor(w)[None,None,:,:],
                                      stride=s, padding=p, dilation=d, groups=g)[0,:,:].tolist()

def checkAnswer(x, y):
    xTensor = torch.tensor(x)
    yTensor = torch.tensor(y)
    if xTensor.shape == yTensor.shape and len(xTensor.shape) == 2:
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i][j] != y[i][j]:
                    return f"The value at index [{i}][{j}] is incorrect."
        return f"{x} is correct!"
    elif len(xTensor.shape) == 2:
        return f"The shape of your answer ({torch.tensor(x).shape}) does not match the expected shape ({torch.tensor(y).shape})"
    else:
        return f"The answer you provided is a {len(xTensor.shape)}D tensor. The expected answer is supposed to be a 2D tensor (matrix)."

def opgave1(answer):
    img = [[1, 2, 1],
           [2, 4, 3],
           [1, 3, 5]]
    weight = [[0, 1],
             [1, 0]]
    expectedAnswer = convolveList(img, weight, 1, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave2(answer):
    img = [[3, 6, 1],
           [5, 0, 1],
           [1, 5, 2]]
    weight = [[2, 3],
              [1, 2]]
    expectedAnswer = convolveList(img, weight, 1, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave3(answer):
    img = [[4, 2, 0, 1, 2],
           [1, 2, 3, 0, 2],
           [0, 4, 1, 2, 3],
           [3, 0, 1, 2, 2]]
    weight = [[ 0, -1,  0],
              [-1,  4, -1],
              [ 0, -1,  0]]
    expectedAnswer = convolveList(img, weight, 1, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave4(answer):
    img = [[1, 3, 2],
           [2, 5, 6],
           [1, 0, 2]]
    weight = [[ 0, -1,  0],
              [-1,  4, -1],
              [ 0, -1,  0]]
    expectedAnswer = convolveList(img, weight, 1, 1, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave5(answer):
    img = [[1, 2],
           [3, 4]]
    weight = [[1, 1],
              [1, 1]]
    expectedAnswer = convolveList(img, weight, 1, 1, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave6(answer):
    img = [[3, 5, 0, 2],
           [4, 3, 2, 0],
           [1, 0, 2, 3],
           [0, 2, 3, 4]]
    weight = [[-1, 1],
              [-1, 1]]
    expectedAnswer = convolveList(img, weight, 2, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave7(answer):
    img = [[3, 5, 0, 2],
           [4, 3, 2, 0],
           [1, 0, 2, 3],
           [0, 2, 3, 4]]
    weight = [[-2, -2],
              [ 2,  2]]
    expectedAnswer = convolveList(img, weight, 2, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave8(answer):
    img = [[2, 5, 7],
           [4, 7, 9],
           [0, 3, 5]]
    weight = [[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]
    expectedAnswer = convolveList(img, weight, 2, 2, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave9(answer):
    img = [[9, 1],
           [7, 3]]
    weight = [[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]
    expectedAnswer = convolveList(img, weight, 2, 2, 1, 1)
    print(checkAnswer(answer, expectedAnswer))
