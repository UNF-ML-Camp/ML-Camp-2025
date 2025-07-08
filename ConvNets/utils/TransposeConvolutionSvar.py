import torch
from random import randint

# Denne funktion bruger pytorch til få den rigtig resultat for transposed convolution operationen
def transConvolveList(x, w, s, p, d, g):
    print(torch.nn.functional.conv_transpose2d(torch.tensor(x).unsqueeze(0), torch.tensor(w)[None,None,:,:],
                                      stride=s, padding=p, dilation=d, groups=g)[0,:,:])
    return torch.nn.functional.conv_transpose2d(torch.tensor(x).unsqueeze(0), torch.tensor(w)[None,None,:,:],
                                      stride=s, padding=p, dilation=d, groups=g)[0,:,:].tolist()

# Denne funktion tjekker om elementerne i to lister er de samme
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

# De her funktioner tjekker en givet svar løser den bestemte opgave
def opgave1(answer):
    img = [[1, 2, 1],
           [2, 4, 3],
           [1, 3, 5]]
    weight = [[0, 1],
             [1, 0]]
    expectedAnswer = transConvolveList(img, weight, 1, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave2(answer):
    img = [[3, 6, 1],
           [5, 0, 1],
           [1, 5, 2]]
    weight = [[2, 3],
              [1, 2]]
    expectedAnswer = transConvolveList(img, weight, 1, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave3(answer):
    img = [[4, 2, 0, 1, 2],
           [1, 2, 3, 0, 2],
           [0, 4, 1, 2, 3],
           [3, 0, 1, 2, 2]]
    weight = [[ 0, -1,  0],
              [-1,  4, -1],
              [ 0, -1,  0]]
    expectedAnswer = transConvolveList(img, weight, 1, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave4(answer):
    img = [[1, 3, 2],
           [2, 5, 6],
           [1, 0, 2]]
    weight = [[ 0, -1,  0],
              [-1,  4, -1],
              [ 0, -1,  0]]
    expectedAnswer = transConvolveList(img, weight, 1, 1, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave5(answer):
    img = [[1, 2],
           [3, 4]]
    weight = [[1, 1],
              [1, 1]]
    expectedAnswer = transConvolveList(img, weight, 1, 1, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave6(answer):
    img = [[3, 5, 0, 2],
           [4, 3, 2, 0],
           [1, 0, 2, 3],
           [0, 2, 3, 4]]
    weight = [[-1, 1],
              [-1, 1]]
    expectedAnswer = transConvolveList(img, weight, 2, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave7(answer):
    img = [[3, 5, 0, 2],
           [4, 3, 2, 0],
           [1, 0, 2, 3],
           [0, 2, 3, 4]]
    weight = [[-2, -2],
              [ 2,  2]]
    expectedAnswer = transConvolveList(img, weight, 2, 0, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave8(answer):
    img = [[2, 5, 7],
           [4, 7, 9],
           [0, 3, 5]]
    weight = [[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]
    expectedAnswer = transConvolveList(img, weight, 2, 2, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

def opgave9(answer):
    img = [[9, 1],
           [7, 3]]
    weight = [[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]
    expectedAnswer = transConvolveList(img, weight, 2, 2, 1, 1)
    print(checkAnswer(answer, expectedAnswer))

# Denne funktion laver en unit test for transposed convolution ved at bruge tilfældige argumenter
# Det her er ikke en komplet unit test, da vi tjekker ikke edge cases
# En edge case kan være en kerne med 0 dimensioner som har stor sandsynlighed for at lave fejl
def createUnitTest(func, stride=False, padding=False):
    # Generer tilfældige argumenter
    imgH = randint(3,10)
    imgW = randint(3,10)
    kernH = randint(1, imgH)
    kernW = randint(1, imgW)
    img = torch.randint(0,10, (imgH, imgW))
    kerne = torch.randint(0,10, (kernH, kernW))
    stride = randint(1,5) if stride else 1
    padding = randint(0,min(kernH, kernW)-1) if padding else 0
    # Compute convolution igennem vores funktion og den rigtigt pytorch funktion
    test = func(img.tolist(), kerne.tolist(), stride, padding)
    correct = torch.nn.functional.conv_transpose2d(img.unsqueeze(0), kerne[None,None,:,:], stride=stride, padding=padding)[0,:,:]
    return torch.allclose(correct, torch.tensor(test))

# Denne funktion tester vores convolution funktion ved at køre en del unit tests
def testFunction(func, stride=False, padding=False):
    # Første laver vi funktionen om for at den altid tager 4 argumenter
    # Det er fordi createUnitTest altid forventer at vores funktion kan tage 4 argumenter
    # Det gør vi med bruge af lambda expressions. I kan google dem, men kort sagt er:
    # "addition = lambda x,y: return x+y" er præcis ligesom at skrive
    # """
    # def addition(x,y):
    #   return x+y
    # """
    if stride and padding:
        newFunc = func
    elif padding:
        newFunc = lambda i,k,s,p: func(i,k,padding=p)
    elif stride:
        newFunc = lambda i,k,s,p: func(i,k,stride=s)
    else:
        newFunc = lambda i,k,s,p: func(i,k)
    # Kør 10 tests uden padding eller stride
    for _ in range(10):
        res = createUnitTest(newFunc)
        if not res:
            print("The function failed to convolve correctly")
            return False
    # Kør 10 tests med stride og uden padding
    if stride:
        for _ in range(10):
            res = createUnitTest(newFunc, stride)
        if not res:
            print("The function failed to convolve correctly when the stride argument was used")
            return False
    # Kør 10 tests med padding og uden stride
    if padding:
        for _ in range(10):
            res = createUnitTest(newFunc, False, padding)
        if not res:
            print("The function failed to convolve correctly when the padding argument was used")
            return False
    # Kør 10 tests med padding og stride
    if stride and padding:
        for _ in range(10):
            res = createUnitTest(newFunc, stride, padding)
        if not res:
            print("The function failed to convolve correctly when both the stride and padding argument was used")
            return False
    print("The function passed all tests")
    return True