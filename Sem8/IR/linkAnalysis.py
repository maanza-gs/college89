import pandas as pd
import numpy as np

class LinkAnalysis:
    def __init__(self,outlinks,alpha,top_k):
        self.outlinks = outlinks
        self.graph = None
        self.adj = None
        self.G = None
        self.pages = None
        self.size = None
        self.alpha = alpha
        self.state_vector = None
        self.iterations = None
        self.eigen = None
        self.k = top_k
        self.hub_score = None
        self.auth_score = None

    def find_no_of_pages(self):
        self.pages = sorted(set(page for link in self.outlinks for page in link))
        self.size = len(self.pages)
        self.state_vector = np.array([1/self.size] * self.size)

    def create_adj_matrix(self):
        df = pd.DataFrame(0, columns=self.pages, index=self.pages)
        for link in self.outlinks:
            df.at[link[0], link[1]] = 1
        self.adj = df
        df = df.div(df.sum(axis=1), axis=0).fillna(0)
        df += ((df.sum(axis=1) == 0).astype(int) / len(self.pages)).values[:, None]
        self.graph = df

    def create_final_matrix(self):
        print(self.alpha)
        self.G = self.alpha * self.graph + (1 - self.alpha) / self.size

    def power_iteration(self):
        iteration=0
        state_vector = self.state_vector
        while True:
            temp = np.dot(state_vector,self.G)
            if np.linalg.norm(temp - state_vector) < 1e-8:
                self.state_vector = temp
                self.iterations = iteration
                return None
            state_vector = temp
            iteration+=1

    def calc_eigen_vector(self,matrix):
        eigen_values, eigen_vectors = np.linalg.eig(matrix)
        principal_eig_val_index = np.argmax(eigen_values)
        ev = eigen_vectors[:, principal_eig_val_index]
        self.eigen = np.divide(ev, np.sum(ev))
        print(self.eigen)

    def calc_hits(self):

        hub_score = np.ones(self.size)
        auth_score = np.ones(self.size)
        max_iterations = 100
        tol = 1e-7
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            new_hub_score = np.dot(self.adj,auth_score)
            new_hub_score /= np.sum(new_hub_score)

            new_auth_score = np.dot(self.adj.T,hub_score)
            new_auth_score /= np.sum(new_auth_score)

            # if np.linalg.norm(new_hub_score - hub_score) < tol and np.linalg.norm(new_auth_score - auth_score) < tol:
            if np.allclose(new_hub_score, hub_score) and np.allclose(new_auth_score, auth_score):
                print(iteration)
                self.hub_score = hub_score
                self.auth_score = auth_score
                return None

            hub_score = new_hub_score
            auth_score = new_auth_score
            # print(hub_score)
            # print(auth_score)

        print(iteration)
        print("Warning: Maximum number of iterations reached without convergence.")
        self.hub_score = hub_score
        self.auth_score = auth_score

    def select_top_k(self,score):
        top_k_pages = np.argsort(score)[::-1][:self.k]
        print(top_k_pages)

    def calc_eigen_hits(self):
        self.calc_eigen_vector(np.dot(self.adj,self.adj.T))
        self.calc_eigen_vector(np.dot(self.adj.T,self.adj))

    def build_link(self):
        self.find_no_of_pages()
        self.create_adj_matrix()
        self.create_final_matrix()
        self.power_iteration()
        self.calc_eigen_vector(self.G.T)
        self.calc_hits()
        self.calc_eigen_hits()

outlinks = [(1,2),(3,2),(2,1),(2,3)]
# outlinks = [
#     ('E', 'F'),
#     ('E', 'C'),
#     ('E', 'D'),
#     ('E', 'B'),
#     ('B', 'E'),
#     ('B', 'C'),
#     ('F', 'C'),
#     ('F', 'H'),
#     ('G', 'A'),
#     ('G', 'C'),
#     ('C', 'A'),
#     ('H', 'A'),
#     ('A', 'D'),
#     ('D', 'B'),
#     ('D', 'C')]
obj = LinkAnalysis(outlinks,0.5,2)
obj.build_link()
print(obj.adj)
print(obj.state_vector)
print(obj.hub_score)
print(obj.auth_score)

obj.select_top_k(obj.state_vector)
obj.select_top_k(obj.auth_score)
obj.select_top_k(obj.hub_score)
