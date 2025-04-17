import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# Function to generate dataset
def generate_data(n, d, sigma=-1):
    X = torch.randn(n, d)
    Y = torch.randn(n, d)
    m = torch.mean(X, dim=0)

    if(sigma != -1):
        m_tilda = m + torch.normal(0, sigma*torch.eye(d))
        return X, Y, m, m_tilda

    return X, Y, m

# Function to find threshold Td
def find_threshold(X, Y, m):
    # Calculate inner products
    inner_products_X = torch.matmul(X, m)
    inner_products_Y = torch.matmul(Y, m)

    # Concatenate results
    all_inner_products = torch.cat([inner_products_X, inner_products_Y])

    # Define a range of thresholds and find the one maximizing accuracy
    thresholds = torch.unique(all_inner_products)
    accuracies = []

    for Td in thresholds:
        predictions_X = (inner_products_X >= Td).to(torch.int)
        predictions_Y = (inner_products_Y < Td).to(torch.int)

        accuracy = torch.cat([predictions_X, predictions_Y]).mean(dtype=float).item()
        accuracies.append(accuracy)

    best_threshold = thresholds[torch.argmax(torch.tensor(accuracies))]

    return best_threshold

# Main experiment
def main_experiment(n, d_range):
    attack_accuracies = []

    for d in d_range:
        # Generate datasets and compute empirical mean
        X, Y, m = generate_data(n, d)

        # Find the threshold Td
        Td = find_threshold(X, Y, m)

        # Simulate the attack using the same datasets
        attack_accuracy = simulate_attack(X, Y, m, Td)
        attack_accuracies.append(attack_accuracy)

    # Plot results
    plt.plot(d_range, attack_accuracies)
    plt.xlabel('Dimension (d)')
    plt.ylabel('Membership Inference Attack Accuracy')
    plt.title('Membership Inference Attack Accuracy vs. Dimension')
    plt.show()

    return Td

# Function to simulate the attack using Td
def simulate_attack(X, Y, m, Td):
    # Simulate the attack using the same datasets
    predictions_X = torch.matmul(X, m) >= Td
    predictions_Y = torch.matmul(Y, m) < Td

    # Calculate accuracy
    accuracy = torch.cat([predictions_X, predictions_Y]).to(torch.float32).mean().item()

    return accuracy

# Run the experiment
n = 50
d_range = range(10, 501)
best_Td = main_experiment(n, d_range)



def experimentTwo(n, d_range, Td):
    attack_accuracies = []

    for d in d_range:
        # Generate datasets and compute empirical mean
        X, Y, m = generate_data(n, d)

        # Simulate the attack using the same datasets
        attack_accuracy = simulate_attack(X, Y, m, Td)
        attack_accuracies.append(attack_accuracy)

    # Plot results
    plt.plot(d_range, attack_accuracies)
    plt.xlabel('Dimension (d)')
    plt.ylabel('Membership Inference Attack Accuracy')
    plt.title('Membership Inference Attack Accuracy vs. Dimension')
    plt.show()

experimentTwo(n, d_range, best_Td)


sigmas = [0, .25, .5, .75, 1]
d = 50

def experimentThree(n, d_range, sigmas):
    attack_accuracies = []
    norms = []

    for sigma in sigmas:
        attack_accuracies_k = []
        norms_k = []
        for i in range(1000):
            # Generate datasets and compute empirical mean
            X, Y, _, m_tilda = generate_data(n, d, sigma)


            # Find the threshold Td
            Td = find_threshold(X, Y, m_tilda)
            # Simulate the attack using the same datasets
            attack_accuracy = simulate_attack(X, Y, m_tilda, Td)
            attack_accuracies_k.append(attack_accuracy)
            norms_k.append(torch.linalg.norm(m_tilda))

        print("done a sigma")
        attack_accuracies.append(np.mean(np.copy(attack_accuracies_k)))
        norms.append(np.mean(np.copy(norms_k)))



    print(attack_accuracies)
    print(norms)

    # Plot results
    plt.plot(sigmas, norms)
    plt.xlabel('Dimension (d)')
    plt.ylabel('l2 Norm')
    plt.title('Means L2-Norm vs. Dimension')
    plt.show()

    # Plot results
    plt.plot(sigmas, attack_accuracies)
    plt.xlabel('Dimension (d)')
    plt.ylabel('Membership Inference Attack Accuracy')
    plt.title('Membership Inference Attack Accuracy vs. Dimension')
    plt.show()

experimentThree(n, d, sigmas)


#PART B)



from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Fashion MNIST dataset
fashion_mnist = datasets.fetch_openml(name="Fashion-MNIST", version=1)
X, y = fashion_mnist.data, fashion_mnist.target
X = X.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

# Set random seed for reproducibility
np.random.seed(42)

# Initialize parameters
n_values = [100, 200, 400, 800, 1600, 2500, 5000, 10000]


def trainer(C=-1):
    train_accuracies = []
    test_accuracies = []
    models = []

    for n in n_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, stratify=y)

        if (C<0):
          model = LogisticRegression(max_iter=5000)
        else:
          model = LogisticRegression(max_iter=5000, C=C)

        model.fit(X_train, y_train)

        # Training accuracy
        train_preds = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        train_accuracies.append(train_accuracy)

        # Testing accuracy
        test_preds = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)
        test_accuracies.append(test_accuracy)

        models.append(model)

    return models, train_accuracies, test_accuracies


unreg_model, unreg_train_accuracies, unreg_test_accuracies = trainer()

# Plot results
plt.figure()
plt.plot(n_values, unreg_train_accuracies, label='Training Accuracy')
plt.plot(n_values, unreg_test_accuracies, label='Testing Accuracy')
plt.xlabel('n')
plt.ylabel('Accuracy')
plt.title('Logistic Regression without Regularization')
plt.legend()
plt.show()


reg_model, reg_train_accuracies, reg_test_accuracies = trainer(0.01)


# Plot results
plt.figure()
plt.plot(n_values, reg_train_accuracies, label='Training Accuracy')
plt.plot(n_values, reg_test_accuracies, label='Testing Accuracy')
plt.xlabel('n')
plt.ylabel('Accuracy')
plt.title('Logistic Regression with L2 Regularization (C=0.01)')
plt.legend()
plt.show

# Continue to Part 3...


index = 0

unreg_attack_accuracies = []
reg_attack_accuracies = []

for n in n_values:

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, stratify=y)

    # Membership inference attack
    attack_preds_unreg = (unreg_model[index].predict(X_train) == y_train).astype(int)
    attack_accuracy_unreg = np.mean(attack_preds_unreg)
    unreg_attack_accuracies.append(attack_accuracy_unreg)

    attack_preds_reg = (reg_model[index].predict(X_train) == y_train).astype(int)
    attack_accuracy_reg = np.mean(attack_preds_reg)
    reg_attack_accuracies.append(attack_accuracy_reg)

    index += 1

# Plot results
plt.figure()
plt.plot(n_values, unreg_attack_accuracies, label='Unregularized Model')
plt.plot(n_values, reg_attack_accuracies, label='L2 Regularized Model')
plt.xlabel('n')
plt.ylabel('Attack Accuracy')
plt.title('Membership Inference Attack Accuracy')
plt.legend()
plt.show()


# Initialize parameters
sigma_values = [0, 1, 2, 3, 4, 5]
defense_accuracies_unreg = []
defense_accuracies_reg = []

# Fixed n = 400
n = 400
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, stratify=y)

# Train logistic regression models for defense
model_unreg_defense = LogisticRegression(max_iter=5000)
model_unreg_defense.fit(X_train, y_train)

model_reg_defense = LogisticRegression(max_iter=5000, C=0.1)
model_reg_defense.fit(X_train, y_train)

# Vary sigma^2 and measure defense accuracy
for sigma in sigma_values:
    # Noisy parameter vectors
    weights_unreg_defense = model_unreg_defense.coef_ + np.random.normal(0, sigma, size=model_unreg_defense.coef_.shape)
    weights_reg_defense = model_reg_defense.coef_ + np.random.normal(0, sigma, size=model_reg_defense.coef_.shape)

    # Set noisy weights to models
    model_unreg_defense.coef_ = weights_unreg_defense
    model_reg_defense.coef_ = weights_reg_defense

    # Defense accuracies
    defense_preds_unreg = model_unreg_defense.predict(X_test) == y_test
    defense_accuracy_unreg = np.mean(defense_preds_unreg)
    defense_accuracies_unreg.append(defense_accuracy_unreg)

    defense_preds_reg = model_reg_defense.predict(X_test) == y_test
    defense_accuracy_reg = np.mean(defense_preds_reg)
    defense_accuracies_reg.append(defense_accuracy_reg)

# Plot results
plt.figure()
plt.plot(sigma_values, defense_accuracies_unreg, label='No Regularization')
plt.xlabel('σ^2')
plt.ylabel('Accuracy')
plt.title('Defense Accuracy - No Regularization')
plt.legend()
plt.show()

plt.figure()
plt.plot(sigma_values, defense_accuracies_reg, label='L2 Regularization (C=0.1)')
plt.xlabel('σ^2')
plt.ylabel('Accuracy')
plt.title('Defense Accuracy - L2 Regularization (C=0.1)')
plt.legend()
plt.show()

# Continue to Part 5...

attack_accuracies_unreg_defense = []
attack_accuracies_reg_defense = []

# Perform membership inference attack after defense
for sigma in sigma_values:
    # Noisy parameter vectors
    weights_unreg_defense = model_unreg_defense.coef_ + np.random.normal(0, sigma, size=model_unreg_defense.coef_.shape)
    weights_reg_defense = model_reg_defense.coef_ + np.random.normal(0, sigma, size=model_reg_defense.coef_.shape)

    # Set noisy weights to models
    model_unreg_defense.coef_ = weights_unreg_defense
    model_reg_defense.coef_ = weights_reg_defense

    # Membership inference attack
    attack_preds_unreg_defense = (model_unreg_defense.predict(X_train) == y_train).astype(int)
    attack_accuracy_unreg_defense = np.mean(attack_preds_unreg_defense)
    attack_accuracies_unreg_defense.append(attack_accuracy_unreg_defense)

    attack_preds_reg_defense = (model_reg_defense.predict(X_train) == y_train).astype(int)
    attack_accuracy_reg_defense = np.mean(attack_preds_reg_defense)
    attack_accuracies_reg_defense.append(attack_accuracy_reg_defense)

# Plot results
plt.figure()
plt.plot(sigma_values, attack_accuracies_unreg_defense, label='Unregularized Model')
plt.plot(sigma_values, attack_accuracies_reg_defense, label='L2 Regularized Model')
plt.xlabel('σ^2')
plt.ylabel('Attack Accuracy')
plt.title('Membership Inference Attack Accuracy')
plt.legend()
plt.show()
