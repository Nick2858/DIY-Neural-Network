import numpy as np
import json
import pickle
import os



class model():
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        rng = np.random.default_rng(1) # random numbers
        self.W1  = rng.normal(size=( input_size, hidden_size))
        self.W2 = rng.normal(size = ( hidden_size, output_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)


    def add_weights(self, model_dict):
        self.W1 = np.array(model_dict["W1"])
        self.W2 = np.array(model_dict["W2"])
        self.b1 = np.array(model_dict["b1"])
        self.b2 = np.array(model_dict["b2"])


    def import_weights(self, path):
        with open(path, "r") as file:
            model_dict = json.load(file)

        self.add_weights(model_dict)


    def inference(self, X , show_hidden = False):
        h = self.relu( X @ self.W1 + self.b1)
        y_hat = self.sigmoid(h @ self.W2 + self.b2 )

        if show_hidden:    
            return h, y_hat

        return y_hat


    def relu(self, z):
        return np.maximum(0,z)
    

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))


    def loss(self, y , y_hat):
        return - np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


    def to_dict(self):
        return {"W1": self.W1.tolist(), "W2": self.W2.tolist(), "b1": self.b1.tolist(), "b2": self.b2.tolist() }



    def export(self, model, path = ""):
        with open(path, "w") as file:
            json.dump(model, file, indent=4, sort_keys=True)


    def train(self, dataset, epoc, err = 0.0001, lr = 1.0, export_path = None):
        X = dataset["inputs"]
        y = dataset["outputs"]
        self.lr = lr
        last_loss = None
        delta_loss = 100

        train = True
        iter = 0

        while train:
            iter += 1
            h, y_hat = self.inference(X, show_hidden=True)

            loss = self.loss(y, y_hat)

            d2 = (y_hat - y ) / len(X)
            d1 = (d2 @ self.W2.T ) * (h > 0)
            
            self.W2 -= self.lr * h.T @ d2
            self.b2 -= self.lr * d2.sum(axis=0)
            self.W1 -= self.lr * X.T @ d1
            self.b1 -= self.lr * d1.sum(axis = 0)

            if iter == 1 or iter %100 == 0:
                print(f"{iter:4d}   {loss:.3f}")

            if last_loss == None:
                last_loss = loss

            else:
                delta_loss = abs(loss - last_loss)
                last_loss = loss
            
            
            if delta_loss < err or iter >= epoc:
                train = False
                print(f"\n{y_hat.round(2)}") # result

        print(f"The model converged in {iter} iterations")
        model_dict = self.to_dict()

        if export_path != None:
            self.export(model_dict, export_path)

        
        return model_dict


# data: 0 if the same, 1 if different
X = np.array([[0.4,0], [1, 2], [0, 1], [1, 0], [0,0], [2.2,2.2], [1,1]])
y = np.array([[1],[1],[1],[1], [0],[0], [0]])

dataset = {"inputs": X, "outputs": y}

# model0 = model(2, 2, 1)
# model_dict = model0.train(dataset= dataset, epoc= 1500, lr=1, err = 0.0000001, export_path="train.json")

# print(model_dict)

model2 = model()
model2.import_weights("train.json")

inference = model2.inference([[1, 1],[0, 2.4], [1, 3]])


print(inference)